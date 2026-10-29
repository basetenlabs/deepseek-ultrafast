use anyhow::{Result, anyhow, bail};
pub use cudarc::driver::sys;
use std::ffi::{CString, c_void};
use std::ptr::{null, null_mut};

pub mod profile;

pub type DevPtr = u64;

#[inline]
pub fn check(r: sys::CUresult, what: &str) -> Result<()> {
    if r == sys::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    let mut s: *const std::ffi::c_char = std::ptr::null();
    unsafe { sys::cuGetErrorString(r, &mut s) };
    let msg = if s.is_null() {
        String::from("?")
    } else {
        unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().into_owned()
    };
    Err(anyhow!("{what}: {r:?} ({msg})"))
}

macro_rules! ck {
    ($e:expr) => {
        check(unsafe { $e }, stringify!($e))
    };
}

pub fn init() -> Result<()> {
    ck!(sys::cuInit(0))
}

pub fn device_cc(ordinal: i32) -> Result<(i32, i32)> {
    let mut dev = 0;
    ck!(sys::cuDeviceGet(&mut dev, ordinal))?;
    let (mut major, mut minor) = (0, 0);
    ck!(sys::cuDeviceGetAttribute(&mut major, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))?;
    ck!(sys::cuDeviceGetAttribute(&mut minor, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev))?;
    Ok((major, minor))
}

pub fn current_attr(a: sys::CUdevice_attribute) -> Result<i32> {
    let mut dev = 0;
    ck!(sys::cuCtxGetDevice(&mut dev))?;
    let mut v = 0;
    ck!(sys::cuDeviceGetAttribute(&mut v, a, dev))?;
    Ok(v)
}

pub struct Device {
    pub ordinal: i32,
    pub dev: sys::CUdevice,
    pub ctx: sys::CUcontext,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub fn current_sm_count() -> Result<i32> {
    let mut dev = 0;
    ck!(sys::cuCtxGetDevice(&mut dev))?;
    let mut v = 0;
    ck!(sys::cuDeviceGetAttribute(&mut v, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev))?;
    Ok(v)
}

impl Device {
    pub fn new(ordinal: i32) -> Result<Self> {
        let mut dev = 0;
        ck!(sys::cuDeviceGet(&mut dev, ordinal))?;
        let mut ctx = null_mut();
        ck!(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev))?;
        let d = Device { ordinal, dev, ctx };
        d.bind()?;
        Ok(d)
    }

    pub fn bind(&self) -> Result<()> {
        ck!(sys::cuCtxSetCurrent(self.ctx))
    }

    pub fn attr(&self, a: sys::CUdevice_attribute) -> Result<i32> {
        let mut v = 0;
        ck!(sys::cuDeviceGetAttribute(&mut v, a, self.dev))?;
        Ok(v)
    }

    pub fn uuid(&self) -> Result<[u8; 16]> {
        let mut uuid = sys::CUuuid { bytes: [0; 16] };
        ck!(sys::cuDeviceGetUuid_v2(&mut uuid, self.dev))?;
        Ok(uuid.bytes.map(|b| b as u8))
    }

    pub fn sm_count(&self) -> Result<i32> {
        self.attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    }

    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok((self.attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?, self.attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?))
    }

    pub fn max_smem_optin(&self) -> Result<i32> {
        self.attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    }

    pub fn max_smem_per_sm(&self) -> Result<i32> {
        self.attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        ck!(sys::cuMemGetInfo_v2(&mut free, &mut total))?;
        Ok((free, total))
    }

    pub fn enable_peer(&self, peer: &Device) -> Result<()> {
        self.bind()?;
        let r = unsafe { sys::cuCtxEnablePeerAccess(peer.ctx, 0) };
        if r == sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            return Ok(());
        }
        check(r, "cuCtxEnablePeerAccess")
    }

    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        ck!(sys::cuCtxSynchronize())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { sys::cuDevicePrimaryCtxRelease_v2(self.dev) };
    }
}

pub struct Buffer {
    pub ptr: DevPtr,
    pub bytes: usize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = 0;
        ck!(sys::cuMemAlloc_v2(&mut ptr, bytes.max(1)))?;
        Ok(Buffer { ptr, bytes })
    }

    pub fn zeroed(bytes: usize) -> Result<Self> {
        let b = Self::alloc(bytes)?;
        ck!(sys::cuMemsetD8_v2(b.ptr, 0, bytes.max(1)))?;
        legacy_fence()?;
        Ok(b)
    }

    pub fn from_host<T: Copy>(data: &[T]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let b = Self::alloc(bytes)?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes > self.bytes {
            bail!("upload of {bytes} B into {} B buffer", self.bytes);
        }
        if bytes == 0 {
            return Ok(());
        }
        ck!(sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, bytes))?;
        legacy_fence()
    }

    pub fn download<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>> {
        download(self.ptr, n)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { sys::cuMemFree_v2(self.ptr) };
    }
}

pub fn download<T: Copy + Default>(ptr: DevPtr, n: usize) -> Result<Vec<T>> {
    let mut v = vec![T::default(); n];
    let bytes = n * std::mem::size_of::<T>();
    if bytes > 0 {
        ck!(sys::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, ptr, bytes))?;
    }
    Ok(v)
}

pub fn upload_synced<T: Copy>(data: &[T]) -> Result<Buffer> {
    thread_local! {
        static STAGE: std::cell::RefCell<Option<PinnedHost>> = const { std::cell::RefCell::new(None) };
    }
    let bytes = std::mem::size_of_val(data);
    let b = Buffer::alloc(bytes)?;
    STAGE.with(|st| -> Result<()> {
        let mut st = st.borrow_mut();
        if st.as_ref().is_none_or(|p| p.bytes < bytes) {
            *st = Some(PinnedHost::alloc(bytes.max(4096))?);
        }
        let p = st.as_ref().unwrap();
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, p.ptr, bytes) };
        ck!(sys::cuMemcpyHtoD_v2(b.ptr, p.ptr as *const c_void, bytes))
    })?;
    Ok(b)
}

pub fn upload<T: Copy>(ptr: DevPtr, data: &[T]) -> Result<()> {
    let bytes = std::mem::size_of_val(data);
    if bytes > 0 {
        ck!(sys::cuMemcpyHtoD_v2(ptr, data.as_ptr() as *const c_void, bytes))?;
        legacy_fence()?;
    }
    Ok(())
}

pub fn legacy_fence() -> Result<()> {
    check(unsafe { sys::cuStreamSynchronize(null_mut()) }, "legacy stream fence")
}

pub fn memset_d8_async(ptr: DevPtr, v: u8, bytes: usize, s: &Stream) -> Result<()> {
    ck!(sys::cuMemsetD8Async(ptr, v, bytes, s.raw))
}

pub fn memset_d32_async(ptr: DevPtr, v: u32, n: usize, s: &Stream) -> Result<()> {
    ck!(sys::cuMemsetD32Async(ptr, v, n, s.raw))
}

pub fn memcpy_dtod_async(dst: DevPtr, src: DevPtr, bytes: usize, s: &Stream) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    ck!(sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, s.raw))
}

pub fn memcpy_htod_async<T: Copy>(dst: DevPtr, src: &[T], s: &Stream) -> Result<()> {
    let bytes = std::mem::size_of_val(src);
    if bytes == 0 {
        return Ok(());
    }
    ck!(sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const c_void, bytes, s.raw))
}

pub fn memcpy_dtoh_async<T: Copy>(dst: &mut [T], src: DevPtr, s: &Stream) -> Result<()> {
    let bytes = std::mem::size_of_val(dst);
    if bytes == 0 {
        return Ok(());
    }
    ck!(sys::cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src, bytes, s.raw))
}

pub struct PinnedHost {
    pub ptr: *mut u8,
    pub bytes: usize,
}

unsafe impl Send for PinnedHost {}
unsafe impl Sync for PinnedHost {}

impl PinnedHost {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let mut p: *mut c_void = null_mut();
        ck!(sys::cuMemAllocHost_v2(&mut p, bytes.max(1)))?;
        Ok(PinnedHost { ptr: p as *mut u8, bytes })
    }
    pub fn as_slice<T>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.bytes / std::mem::size_of::<T>()) }
    }
    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.bytes / std::mem::size_of::<T>()) }
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        unsafe { sys::cuMemFreeHost(self.ptr as *mut c_void) };
    }
}

pub struct Stream {
    pub raw: sys::CUstream,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut raw = null_mut();
        ck!(sys::cuStreamCreate(&mut raw, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32))?;
        Ok(Stream { raw })
    }
    pub fn with_priority(prio: i32) -> Result<Self> {
        let mut raw = null_mut();
        ck!(sys::cuStreamCreateWithPriority(&mut raw, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32, prio))?;
        Ok(Stream { raw })
    }
    pub fn priority_range() -> Result<(i32, i32)> {
        let (mut lo, mut hi) = (0, 0);
        ck!(sys::cuCtxGetStreamPriorityRange(&mut lo, &mut hi))?;
        Ok((lo, hi))
    }
    pub fn synchronize(&self) -> Result<()> {
        ck!(sys::cuStreamSynchronize(self.raw))
    }
    pub fn wait(&self, e: &Event) -> Result<()> {
        ck!(sys::cuStreamWaitEvent(self.raw, e.raw, 0))
    }
    pub fn is_capturing(&self) -> Result<bool> {
        let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        ck!(sys::cuStreamIsCapturing(self.raw, &mut st))?;
        Ok(st == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
    }
    pub fn begin_capture(&self) -> Result<()> {
        ck!(sys::cuStreamBeginCapture_v2(
            self.raw,
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        ))
    }
    pub fn end_capture(&self) -> Result<Graph> {
        let mut g = null_mut();
        ck!(sys::cuStreamEndCapture(self.raw, &mut g))?;
        Ok(Graph { raw: g })
    }
    pub fn begin_capture_into(&self, g: sys::CUgraph) -> Result<()> {
        ck!(sys::cuStreamBeginCaptureToGraph(self.raw, g, null(), null(), 0, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
    }
    pub fn end_capture_into(&self) -> Result<()> {
        let mut g = null_mut();
        ck!(sys::cuStreamEndCapture(self.raw, &mut g))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { sys::cuStreamDestroy_v2(self.raw) };
    }
}

pub struct Event {
    pub raw: sys::CUevent,
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        let mut raw = null_mut();
        ck!(sys::cuEventCreate(&mut raw, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32))?;
        Ok(Event { raw })
    }
    pub fn timed() -> Result<Self> {
        let mut raw = null_mut();
        ck!(sys::cuEventCreate(&mut raw, sys::CUevent_flags::CU_EVENT_DEFAULT as u32))?;
        Ok(Event { raw })
    }
    pub fn record(&self, s: &Stream) -> Result<()> {
        ck!(sys::cuEventRecord(self.raw, s.raw))
    }
    pub fn synchronize(&self) -> Result<()> {
        ck!(sys::cuEventSynchronize(self.raw))
    }
    pub fn elapsed_ms(&self, later: &Event) -> Result<f32> {
        let mut ms = 0f32;
        ck!(sys::cuEventElapsedTime_v2(&mut ms, self.raw, later.raw))?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { sys::cuEventDestroy_v2(self.raw) };
    }
}

pub struct Graph {
    pub raw: sys::CUgraph,
}

unsafe impl Send for Graph {}

impl Graph {
    pub fn instantiate(&self) -> Result<GraphExec> {
        let mut e = null_mut();
        let flags = if crate::profile::sm103() { 8 } else { 0 };
        ck!(sys::cuGraphInstantiateWithFlags(&mut e, self.raw, flags))?;
        Ok(GraphExec { raw: e })
    }
    pub fn node_count(&self) -> Result<usize> {
        let mut n = 0usize;
        ck!(sys::cuGraphGetNodes(self.raw, null_mut(), &mut n))?;
        Ok(n)
    }
}

impl Graph {
    pub fn empty() -> Result<Graph> {
        let mut g = null_mut();
        ck!(sys::cuGraphCreate(&mut g, 0))?;
        Ok(Graph { raw: g })
    }
    pub fn cond_handle_flags(&self, default: u32, assign_default: bool) -> Result<u64> {
        let (mut h, mut ctx) = (0u64, null_mut());
        ck!(sys::cuCtxGetCurrent(&mut ctx))?;
        ck!(sys::cuGraphConditionalHandleCreate(&mut h, self.raw, ctx, default, assign_default as u32))?;
        Ok(h)
    }
    pub fn sinks(&self) -> Result<Vec<sys::CUgraphNode>> {
        let mut n = 0usize;
        ck!(sys::cuGraphGetNodes(self.raw, null_mut(), &mut n))?;
        let mut nodes = vec![null_mut(); n];
        ck!(sys::cuGraphGetNodes(self.raw, nodes.as_mut_ptr(), &mut n))?;
        let mut out = vec![];
        for nd in nodes {
            let mut d = 0usize;
            ck!(sys::cuGraphNodeGetDependentNodes_v2(nd, null_mut(), null_mut(), &mut d))?;
            if d == 0 {
                out.push(nd);
            }
        }
        Ok(out)
    }
    pub fn add_switch(&self, deps: &[sys::CUgraphNode], h: u64, n: u32) -> Result<Vec<sys::CUgraph>> {
        let mut ctx = null_mut();
        ck!(sys::cuCtxGetCurrent(&mut ctx))?;
        let mut p: sys::CUgraphNodeParams = unsafe { std::mem::zeroed() };
        p.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
        p.__bindgen_anon_1.conditional = sys::CUDA_CONDITIONAL_NODE_PARAMS {
            handle: h,
            type_: sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_SWITCH,
            size: n,
            phGraph_out: null_mut(),
            ctx,
        };
        let mut node = null_mut();
        ck!(sys::cuGraphAddNode_v2(&mut node, self.raw, deps.as_ptr(), null(), deps.len(), &mut p))?;
        let bodies = unsafe { std::slice::from_raw_parts(p.__bindgen_anon_1.conditional.phGraph_out, n as usize) }.to_vec();
        Ok(bodies)
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { sys::cuGraphDestroy(self.raw) };
    }
}

pub struct GraphExec {
    pub raw: sys::CUgraphExec,
}

unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    pub fn upload(&self, s: &Stream) -> Result<()> {
        ck!(sys::cuGraphUpload(self.raw, s.raw))
    }
    pub fn launch(&self, s: &Stream) -> Result<()> {
        ck!(sys::cuGraphLaunch(self.raw, s.raw))
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        unsafe { sys::cuGraphExecDestroy(self.raw) };
    }
}

pub struct Module {
    pub raw: sys::CUmodule,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    pub fn load(image: &[u8]) -> Result<Self> {
        let mut owned;
        let data = if image.first() == Some(&0x7f) {
            image.as_ptr()
        } else {
            owned = image.to_vec();
            owned.push(0);
            owned.as_ptr()
        };
        let mut m = null_mut();
        ck!(sys::cuModuleLoadData(&mut m, data as *const c_void))?;
        Ok(Module { raw: m })
    }

    pub fn function(&self, name: &str) -> Result<Function> {
        let c = CString::new(name)?;
        let mut f = null_mut();
        check(
            unsafe { sys::cuModuleGetFunction(&mut f, self.raw, c.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(Function { raw: f, name: name.to_string() })
    }

    pub fn global(&self, name: &str) -> Result<(DevPtr, usize)> {
        let c = CString::new(name)?;
        let (mut p, mut n) = (0, 0usize);
        ck!(sys::cuModuleGetGlobal_v2(&mut p, &mut n, self.raw, c.as_ptr()))?;
        Ok((p, n))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { sys::cuModuleUnload(self.raw) };
    }
}

#[derive(Clone)]
pub struct Function {
    pub raw: sys::CUfunction,
    pub name: String,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    pub fn set_max_dynamic_smem(&self, bytes: u32) -> Result<()> {
        ck!(sys::cuFuncSetAttribute(
            self.raw,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32
        ))
    }
    pub fn set_carveout(&self, pct: i32) -> Result<()> {
        ck!(sys::cuFuncSetAttribute(
            self.raw,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            pct
        ))
    }
    pub fn set_non_portable_cluster(&self) -> Result<()> {
        ck!(sys::cuFuncSetAttribute(
            self.raw,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
            1
        ))
    }
    pub fn attr(&self, a: sys::CUfunction_attribute) -> Result<i32> {
        let mut v = 0;
        ck!(sys::cuFuncGetAttribute(&mut v, a, self.raw))?;
        Ok(v)
    }
    pub fn max_active_blocks_per_sm(&self, threads: u32, smem: usize) -> Result<i32> {
        let mut n = 0;
        ck!(sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut n, self.raw, threads as i32, smem))?;
        Ok(n)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Launch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
    pub pdl: bool,
    pub cluster: Option<(u32, u32, u32)>,
}

impl Launch {
    pub fn new(grid: u32, block: u32) -> Self {
        Launch { grid: (grid, 1, 1), block: (block, 1, 1), smem: 0, pdl: false, cluster: None }
    }
    pub fn grid3(mut self, g: (u32, u32, u32)) -> Self {
        self.grid = g;
        self
    }
    pub fn smem(mut self, s: u32) -> Self {
        self.smem = s;
        self
    }
    pub fn pdl(mut self, on: bool) -> Self {
        self.pdl = on;
        self
    }
}

#[derive(Default)]
pub struct Args {
    slots: Vec<Box<[u64]>>,
}

impl Args {
    pub fn new() -> Self {
        Args { slots: Vec::with_capacity(16) }
    }
    pub fn push<T: Copy>(&mut self, v: T) -> &mut Self {
        let n = std::mem::size_of::<T>().div_ceil(8).max(1);
        let mut slot = vec![0u64; n].into_boxed_slice();
        unsafe {
            std::ptr::copy_nonoverlapping(&v as *const T as *const u8, slot.as_mut_ptr() as *mut u8, std::mem::size_of::<T>())
        };
        self.slots.push(slot);
        self
    }
    pub fn ptr(&mut self, p: DevPtr) -> &mut Self {
        self.push(p)
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.push(v)
    }
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.push(v)
    }
    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.push(v)
    }
}

pub fn launch(f: &Function, cfg: &Launch, stream: &Stream, args: &mut Args) -> Result<()> {
    let mut ptrs: Vec<*mut c_void> = args.slots.iter_mut().map(|s| s.as_mut_ptr() as *mut c_void).collect();
    let mut attrs: Vec<sys::CUlaunchAttribute> = Vec::new();
    if cfg.pdl {
        let mut a: sys::CUlaunchAttribute = unsafe { std::mem::zeroed() };
        a.id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        a.value.programmaticStreamSerializationAllowed = 1;
        attrs.push(a);
    }
    if let Some((x, y, z)) = cfg.cluster {
        let mut a: sys::CUlaunchAttribute = unsafe { std::mem::zeroed() };
        a.id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        a.value.clusterDim.x = x;
        a.value.clusterDim.y = y;
        a.value.clusterDim.z = z;
        attrs.push(a);
    }
    let (attr_ptr, nattr) = (if attrs.is_empty() { null_mut() } else { attrs.as_mut_ptr() }, attrs.len());
    let lc = sys::CUlaunchConfig {
        gridDimX: cfg.grid.0,
        gridDimY: cfg.grid.1,
        gridDimZ: cfg.grid.2,
        blockDimX: cfg.block.0,
        blockDimY: cfg.block.1,
        blockDimZ: cfg.block.2,
        sharedMemBytes: cfg.smem,
        hStream: stream.raw,
        attrs: attr_ptr,
        numAttrs: nattr as u32,
    };
    check(
        unsafe { sys::cuLaunchKernelEx(&lc, f.raw, ptrs.as_mut_ptr(), null_mut()) },
        &format!("launch {}", f.name),
    )
}

pub fn max_active_clusters(f: &Function, cfg: &Launch) -> Result<i32> {
    let mut attrs: Vec<sys::CUlaunchAttribute> = Vec::new();
    if let Some((x, y, z)) = cfg.cluster {
        let mut a: sys::CUlaunchAttribute = unsafe { std::mem::zeroed() };
        a.id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        a.value.clusterDim.x = x;
        a.value.clusterDim.y = y;
        a.value.clusterDim.z = z;
        attrs.push(a);
    }
    let (attr_ptr, nattr) = (if attrs.is_empty() { null_mut() } else { attrs.as_mut_ptr() }, attrs.len());
    let lc = sys::CUlaunchConfig {
        gridDimX: cfg.grid.0,
        gridDimY: cfg.grid.1,
        gridDimZ: cfg.grid.2,
        blockDimX: cfg.block.0,
        blockDimY: cfg.block.1,
        blockDimZ: cfg.block.2,
        sharedMemBytes: cfg.smem,
        hStream: null_mut(),
        attrs: attr_ptr,
        numAttrs: nattr as u32,
    };
    let mut n = 0;
    ck!(sys::cuOccupancyMaxActiveClusters(&mut n, f.raw, &lc))?;
    Ok(n)
}

pub fn tensor_map_fp4_unpack_strided(base: DevPtr, d0: u64, d1: u64, d2: u64, d2_stride: u64, box_rows: u32) -> Result<[u64; 16]> {
    let mut m: sys::CUtensorMap = unsafe { std::mem::zeroed() };
    let dims = [d0, d1, d2];
    let strides = [d0 / 2, d2_stride];
    let boxd = [128u32, box_rows, 1];
    let es = [1u32, 1, 1];
    ck!(sys::cuTensorMapEncodeTiled(
        &mut m,
        sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        3,
        base as *mut c_void,
        dims.as_ptr(),
        strides.as_ptr(),
        boxd.as_ptr(),
        es.as_ptr(),
        sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ))?;
    Ok(m.opaque)
}

#[allow(clippy::too_many_arguments)]
pub fn tensor_map_fp4_unpack_extent(base: DevPtr, d0: u64, d1: u64, d2: u64, row_bytes: u64, d2_stride: u64, box_rows: u32) -> Result<[u64; 16]> {
    let mut m: sys::CUtensorMap = unsafe { std::mem::zeroed() };
    let dims = [d0, d1, d2];
    let strides = [row_bytes, d2_stride];
    let boxd = [128u32, box_rows, 1];
    let es = [1u32, 1, 1];
    ck!(sys::cuTensorMapEncodeTiled(
        &mut m,
        sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        3,
        base as *mut c_void,
        dims.as_ptr(),
        strides.as_ptr(),
        boxd.as_ptr(),
        es.as_ptr(),
        sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ))?;
    Ok(m.opaque)
}

pub fn tensor_map_u8_sw128(base: DevPtr, cols: u64, rows: u64, stride: u64, box_rows: u32) -> Result<[u64; 16]> {
    let mut m: sys::CUtensorMap = unsafe { std::mem::zeroed() };
    let dims = [cols, rows];
    let strides = [stride];
    let boxd = [128u32, box_rows];
    let es = [1u32, 1];
    ck!(sys::cuTensorMapEncodeTiled(
        &mut m,
        sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        base as *mut c_void,
        dims.as_ptr(),
        strides.as_ptr(),
        boxd.as_ptr(),
        es.as_ptr(),
        sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ))?;
    Ok(m.opaque)
}

pub fn tensor_map_fp4_rows(base: DevPtr, rows: u64, box_rows: u32) -> Result<[u64; 16]> {
    let mut m: sys::CUtensorMap = unsafe { std::mem::zeroed() };
    let dims = [16u64, rows, 4];
    let strides = [64u64, 16];
    let boxd = [16u32, box_rows, 4];
    let es = [1u32, 1, 1];
    ck!(sys::cuTensorMapEncodeTiled(
        &mut m,
        sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        base as *mut c_void,
        dims.as_ptr(),
        strides.as_ptr(),
        boxd.as_ptr(),
        es.as_ptr(),
        sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ))?;
    Ok(m.opaque)
}

pub fn tensor_map_bf16_sw128(base: DevPtr, cols: u64, rows: u64, stride: u64, box_rows: u32) -> Result<[u64; 16]> {
    let mut m: sys::CUtensorMap = unsafe { std::mem::zeroed() };
    let dims = [cols, rows];
    let strides = [stride];
    let boxd = [64u32, box_rows];
    let es = [1u32, 1];
    ck!(sys::cuTensorMapEncodeTiled(
        &mut m,
        sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        base as *mut c_void,
        dims.as_ptr(),
        strides.as_ptr(),
        boxd.as_ptr(),
        es.as_ptr(),
        sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ))?;
    Ok(m.opaque)
}

pub struct MappedHost {
    pub host: *mut u8,
    pub dev: DevPtr,
    pub bytes: usize,
}

unsafe impl Send for MappedHost {}
unsafe impl Sync for MappedHost {}

impl MappedHost {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let mut p: *mut c_void = null_mut();
        ck!(sys::cuMemHostAlloc(&mut p, bytes, 0x02 ))?;
        let mut d: DevPtr = 0;
        ck!(sys::cuMemHostGetDevicePointer_v2(&mut d, p, 0))?;
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
        Ok(MappedHost { host: p as *mut u8, dev: d, bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PTX: &str = ".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry k(.param .u64 p)\n{\n.reg .b64 %rd<3>;\n.reg .b32 %r<2>;\nld.param.u64 %rd1, [p];\ncvta.to.global.u64 %rd2, %rd1;\nmov.u32 %r1, 7;\nst.global.u32 [%rd2], %r1;\nret;\n}\n";

    #[test]
    #[ignore]
    fn launch_on_device() -> Result<()> {
        init()?;
        let d = Device::new(0)?;
        println!("cc {:?} sms {} optin {} per-sm {}", d.compute_capability()?, d.sm_count()?, d.max_smem_optin()?, d.max_smem_per_sm()?);
        let m = Module::load(PTX.as_bytes())?;
        let f = m.function("k")?;
        let out = Buffer::zeroed(4)?;
        let s = Stream::new()?;
        let cfg = Launch::new(1, 32).pdl(true);
        launch(&f, &cfg, &s, Args::new().ptr(out.ptr))?;
        s.synchronize()?;
        assert_eq!(out.download::<u32>(1)?[0], 7);
        let mut one = cfg;
        one.cluster = Some((1, 1, 1));
        println!("blocks/SM {} co-resident CTAs (cluster 1) {:?}", f.max_active_blocks_per_sm(32, 0)?, max_active_clusters(&f, &one).map_err(|e| e.to_string()));
        Ok(())
    }
}

use anyhow::{Result, ensure, Context};
use deepseek_ultrafast_cuda::{Buffer, DevPtr, Stream};
use cudarc::cudnn::sys::{self as d, cudnnBackendDescriptorType_t::*, cudnnBackendAttributeName_t::*,
    cudnnBackendAttributeType_t::*, cudnnDataType_t::*, cudnnPointwiseMode_t::*, cudnnReduceTensorOp_t::*};
use std::{ffi::c_void, collections::VecDeque};
fn check(s: d::cudnnStatus_t) -> Result<()> { ensure!(s == d::cudnnStatus_t::CUDNN_STATUS_SUCCESS, "cuDNN vision: {s:?}"); Ok(()) }
struct Descriptor(d::cudnnBackendDescriptor_t);
impl Descriptor {
    fn new(ty: d::cudnnBackendDescriptorType_t) -> Result<Self> {
        let mut p = std::ptr::null_mut(); check(unsafe { d::cudnnBackendCreateDescriptor(ty, &mut p) })?; Ok(Self(p))
    }
    fn set<T>(&self, key: d::cudnnBackendAttributeName_t, ty: d::cudnnBackendAttributeType_t, values: &[T]) -> Result<()> {
        check(unsafe { d::cudnnBackendSetAttribute(self.0, key, ty, values.len() as i64, values.as_ptr().cast()) }).with_context(|| format!("set {key:?}"))
    }
    fn finish(self) -> Result<Self> { check(unsafe { d::cudnnBackendFinalize(self.0) })?; Ok(self) }
    fn children(&self, key: d::cudnnBackendAttributeName_t, kind: d::cudnnBackendDescriptorType_t, capacity: usize) -> Result<Vec<Self>> {
        let mut children = (0..capacity).map(|_| Self::new(kind)).collect::<Result<Vec<_>>>()?;
        let mut pointers: Vec<_> = children.iter().map(|c| c.0).collect();
        let mut count = 0;
        check(unsafe { d::cudnnBackendGetAttribute(self.0, key, CUDNN_TYPE_BACKEND_DESCRIPTOR,
            capacity as i64, &mut count, pointers.as_mut_ptr().cast()) })?;
        ensure!(count >= 0 && count as usize <= capacity, "cuDNN descriptor result exceeds capacity");
        children.truncate(count as usize);
        Ok(children)
    }
    fn i64(&self, key: d::cudnnBackendAttributeName_t) -> Result<i64> {
        let (mut n, mut value) = (0,0i64);
        check(unsafe { d::cudnnBackendGetAttribute(self.0,key,CUDNN_TYPE_INT64,1,&mut n,(&mut value as *mut i64).cast()) })?;
        ensure!(n == 1); Ok(value)
    }
}
impl Drop for Descriptor { fn drop(&mut self) { unsafe { d::cudnnBackendDestroyDescriptor(self.0); } } }
fn excluded_sdpa_config(version: usize, engine: i64, kernel: i64) -> bool {
    version == 91400 && engine == 3 && kernel == 36
}
fn has_sdpa_erratum(config: &Descriptor) -> Result<bool> {
    let version = unsafe { d::cudnnGetVersion() };
    if version != 91400 { return Ok(false); }
    let engine = config.children(CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_BACKEND_ENGINE_DESCRIPTOR, 1)?;
    ensure!(engine.len() == 1, "cuDNN configuration has no engine");
    let index = engine[0].i64(CUDNN_ATTR_ENGINE_GLOBAL_INDEX)?;
    if index != 3 { return Ok(false); }
    for knob in config.children(CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, 32)? {
        let (mut kind, mut count) = (0i32, 0);
        check(unsafe { d::cudnnBackendGetAttribute(knob.0, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
            CUDNN_TYPE_KNOB_TYPE, 1, &mut count, (&mut kind as *mut i32).cast()) })?;
        ensure!(count == 1, "cuDNN knob has no type");
        if kind == d::cudnnBackendKnobType_t::CUDNN_KNOB_TYPE_KERNEL_CFG as i32
            && excluded_sdpa_config(version, index, knob.i64(CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE)?) {
            return Ok(true);
        }
    }
    Ok(false)
}
struct Handle(d::cudnnHandle_t);
impl Drop for Handle { fn drop(&mut self) { unsafe { d::cudnnDestroy(self.0); } } }
struct Plan {
    n: usize, plan: Descriptor, workspace: Buffer,
    pack: Option<(DevPtr,DevPtr,Descriptor)>,
    scale: Box<f32>,
    _descriptors: Vec<Descriptor>,
}
struct Builder { descriptors: Vec<Descriptor>, ops: Vec<d::cudnnBackendDescriptor_t> }
impl Builder {
    fn tensor(&mut self, id:i64, dims:[i64;4], strides:[i64;4], bf16:bool, virt:bool, by_value:bool) -> Result<d::cudnnBackendDescriptor_t> {
        let t = Descriptor::new(CUDNN_BACKEND_TENSOR_DESCRIPTOR)?;
        t.set(CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,&[if bf16 { CUDNN_DATA_BFLOAT16 } else { CUDNN_DATA_FLOAT }])?;
        t.set(CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,&dims)?; t.set(CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,&strides)?;
        t.set(CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,&[id])?; t.set(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,&[16i64])?;
        t.set(CUDNN_ATTR_TENSOR_IS_VIRTUAL,CUDNN_TYPE_BOOLEAN,&[virt])?;
        if by_value { t.set(CUDNN_ATTR_TENSOR_IS_BY_VALUE,CUDNN_TYPE_BOOLEAN,&[true])?; }
        let t=t.finish()?; let p=t.0; self.descriptors.push(t); Ok(p)
    }
    fn matmul(&mut self, a:d::cudnnBackendDescriptor_t,b:d::cudnnBackendDescriptor_t,c:d::cudnnBackendDescriptor_t) -> Result<()> {
        let desc=Descriptor::new(CUDNN_BACKEND_MATMUL_DESCRIPTOR)?;
        desc.set(CUDNN_ATTR_MATMUL_COMP_TYPE,CUDNN_TYPE_DATA_TYPE,&[CUDNN_DATA_FLOAT])?;
        let desc=desc.finish()?; let op=Descriptor::new(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)?;
        for (attr,p) in [(CUDNN_ATTR_OPERATION_MATMUL_ADESC,a),(CUDNN_ATTR_OPERATION_MATMUL_BDESC,b),(CUDNN_ATTR_OPERATION_MATMUL_CDESC,c),(CUDNN_ATTR_OPERATION_MATMUL_DESC,desc.0)] {
            op.set(attr,CUDNN_TYPE_BACKEND_DESCRIPTOR,&[p])?;
        }
        let op=op.finish()?; self.ops.push(op.0); self.descriptors.extend([desc,op]); Ok(())
    }
    fn point(&mut self, mode:d::cudnnPointwiseMode_t,a:d::cudnnBackendDescriptor_t,b:Option<d::cudnnBackendDescriptor_t>,c:d::cudnnBackendDescriptor_t) -> Result<()> {
        let desc=Descriptor::new(CUDNN_BACKEND_POINTWISE_DESCRIPTOR)?;
        desc.set(CUDNN_ATTR_POINTWISE_MODE,CUDNN_TYPE_POINTWISE_MODE,&[mode])?;
        desc.set(CUDNN_ATTR_POINTWISE_MATH_PREC,CUDNN_TYPE_DATA_TYPE,&[CUDNN_DATA_FLOAT])?;
        let desc=desc.finish()?; let op=Descriptor::new(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)?;
        for (attr,p) in [(CUDNN_ATTR_OPERATION_POINTWISE_XDESC,a),(CUDNN_ATTR_OPERATION_POINTWISE_YDESC,c),(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,desc.0)] {
            op.set(attr,CUDNN_TYPE_BACKEND_DESCRIPTOR,&[p])?;
        }
        if let Some(b)=b { op.set(CUDNN_ATTR_OPERATION_POINTWISE_BDESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,&[b])?; }
        let op=op.finish()?; self.ops.push(op.0); self.descriptors.extend([desc,op]); Ok(())
    }
    fn reduce(&mut self, mode:d::cudnnReduceTensorOp_t,a:d::cudnnBackendDescriptor_t,c:d::cudnnBackendDescriptor_t) -> Result<()> {
        let desc=Descriptor::new(CUDNN_BACKEND_REDUCTION_DESCRIPTOR)?;
        desc.set(CUDNN_ATTR_REDUCTION_OPERATOR,CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,&[mode])?;
        desc.set(CUDNN_ATTR_REDUCTION_COMP_TYPE,CUDNN_TYPE_DATA_TYPE,&[CUDNN_DATA_FLOAT])?;
        let desc=desc.finish()?; let op=Descriptor::new(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)?;
        for (attr,p) in [(CUDNN_ATTR_OPERATION_REDUCTION_XDESC,a),(CUDNN_ATTR_OPERATION_REDUCTION_YDESC,c),(CUDNN_ATTR_OPERATION_REDUCTION_DESC,desc.0)] {
            op.set(attr,CUDNN_TYPE_BACKEND_DESCRIPTOR,&[p])?;
        }
        let op=op.finish()?; self.ops.push(op.0); self.descriptors.extend([desc,op]); Ok(())
    }
}
impl Plan {
    fn new(n:usize,handle:d::cudnnHandle_t) -> Result<Self> {
        let s=n as i64; let mut b=Builder { descriptors:Vec::new(),ops:Vec::new() };
        let q=b.tensor(1,[1,16,s,64],[s*3072,64,3072,1],true,false,false)?;
        let k=b.tensor(2,[1,16,64,s],[s*3072,64,1,3072],true,false,false)?;
        let v=b.tensor(3,[1,16,s,64],[s*3072,64,3072,1],true,false,false)?;
        let o=b.tensor(4,[1,16,s,64],[s*1024,64,1024,1],true,false,false)?;
        let scale=b.tensor(5,[1,1,1,1],[1,1,1,1],false,false,true)?;
        let score=b.tensor(6,[1,16,s,s],[16*s*s,s*s,s,1],false,true,false)?;
        let scaled=b.tensor(7,[1,16,s,s],[16*s*s,s*s,s,1],false,true,false)?;
        let max=b.tensor(8,[1,16,s,1],[16*s,s,1,1],false,true,false)?;
        let sub=b.tensor(9,[1,16,s,s],[16*s*s,s*s,s,1],false,true,false)?;
        let exp=b.tensor(10,[1,16,s,s],[16*s*s,s*s,s,1],false,true,false)?;
        let sum=b.tensor(11,[1,16,s,1],[16*s,s,1,1],false,true,false)?;
        let prob=b.tensor(12,[1,16,s,s],[16*s*s,s*s,s,1],true,true,false)?;
        b.matmul(q,k,score)?; b.point(CUDNN_POINTWISE_MUL,score,Some(scale),scaled)?;
        b.reduce(CUDNN_REDUCE_TENSOR_MAX,scaled,max)?; b.point(CUDNN_POINTWISE_SUB,scaled,Some(max),sub)?;
        b.point(CUDNN_POINTWISE_EXP,sub,None,exp)?; b.reduce(CUDNN_REDUCE_TENSOR_ADD,exp,sum)?;
        let log=b.tensor(13,[1,16,s,1],[16*s,s,1,1],false,true,false)?;
        let stats=b.tensor(14,[1,16,s,1],[16*s,s,1,1],false,true,false)?;
        b.point(CUDNN_POINTWISE_LOG,sum,None,log)?; b.point(CUDNN_POINTWISE_ADD,max,Some(log),stats)?;
        b.point(CUDNN_POINTWISE_DIV,exp,Some(sum),prob)?; b.matmul(prob,v,o)?;
        let graph=Descriptor::new(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)?;
        graph.set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE,CUDNN_TYPE_HANDLE,&[handle])?;
        graph.set(CUDNN_ATTR_OPERATIONGRAPH_OPS,CUDNN_TYPE_BACKEND_DESCRIPTOR,&b.ops)?;
        let graph=graph.finish()?;
        let heur=Descriptor::new(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)?;
        heur.set(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,CUDNN_TYPE_BACKEND_DESCRIPTOR,&[graph.0])?;
        heur.set(CUDNN_ATTR_ENGINEHEUR_MODE,CUDNN_TYPE_HEUR_MODE,&[d::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A])?;
        let heur=heur.finish()?;
        let configs:Vec<_>=(0..16).map(|_| Descriptor::new(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)).collect::<Result<_>>()?;
        let mut ptrs:Vec<_>=configs.iter().map(|c|c.0).collect(); let mut count=0;
        check(unsafe { d::cudnnBackendGetAttribute(heur.0,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,16,&mut count,ptrs.as_mut_ptr().cast()) })?;
        for config in configs.iter().take((count as usize).min(16)) {
            if has_sdpa_erratum(config)? { continue; }
            let plan=Descriptor::new(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)?;
            plan.set(CUDNN_ATTR_EXECUTION_PLAN_HANDLE,CUDNN_TYPE_HANDLE,&[handle])?;
            plan.set(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,CUDNN_TYPE_BACKEND_DESCRIPTOR,&[config.0])?;
            let Ok(plan)=plan.finish() else { continue; };
            let size=plan.i64(CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)?;
            if !(0..=128*1024*1024).contains(&size) { continue; }
            let workspace=Buffer::alloc((size as usize).max(1))?;
            b.descriptors.extend([graph,heur]); b.descriptors.extend(configs);
            return Ok(Self { n,plan,workspace,pack:None,scale:Box::new(0.125),_descriptors:b.descriptors });
        }
        anyhow::bail!("no bounded non-excluded fused cuDNN vision attention plan for {n} patches")
    }
    fn run(&mut self, handle:d::cudnnHandle_t,qkv:DevPtr,out:DevPtr) -> Result<()> {
        if self.pack.as_ref().is_none_or(|(q,o,_)| *q != qkv || *o != out) {
            let pack=Descriptor::new(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)?;
            let pointers=[qkv as *mut c_void,(qkv+2048) as _,(qkv+4096) as _,out as _,(&mut *self.scale as *mut f32).cast()];
            pack.set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,CUDNN_TYPE_INT64,&[1i64,2,3,4,5])?;
            pack.set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,CUDNN_TYPE_VOID_PTR,&pointers)?;
            pack.set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE,CUDNN_TYPE_VOID_PTR,&[self.workspace.ptr as *mut c_void])?;
            self.pack=Some((qkv,out,pack.finish()?));
        }
        check(unsafe { d::cudnnBackendExecute(handle,self.plan.0,self.pack.as_ref().unwrap().2.0) })
    }
}
pub struct Attention { plans:VecDeque<Plan>, handle:Handle }
impl Attention {
    pub fn new(stream:&Stream) -> Result<Self> {
        ensure!(unsafe { d::cudnnGetVersion() } >= 91000, "image serving needs cuDNN >=9.10");
        let mut h=std::ptr::null_mut(); check(unsafe { d::cudnnCreate(&mut h) })?;
        let handle=Handle(h); check(unsafe { d::cudnnSetStream(h,stream.raw as _) })?;
        Ok(Self { plans:VecDeque::new(),handle })
    }
    pub fn run(&mut self,n:usize,qkv:DevPtr,out:DevPtr) -> Result<()> {
        let at=if let Some(at)=self.plans.iter().position(|p|p.n==n) { at } else {
            let p=Plan::new(n,self.handle.0)?;
            if self.plans.len()==4 { self.plans.pop_front(); }
            self.plans.push_back(p); self.plans.len()-1
        };
        self.plans[at].run(self.handle.0,qkv,out)
    }
}

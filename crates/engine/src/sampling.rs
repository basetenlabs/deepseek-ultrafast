use deepseek_ultrafast_cuda::DevPtr;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SampParams {
    pub temp: f32,
    pub top_p: f32,
    pub seed: u64,
    pub step: u64,
    pub _pad: u32,
}
const _: () = assert!(std::mem::size_of::<SampParams>() == 32);

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SampIo {
    pub logits: DevPtr,
    pub vloc: i32,
    pub vpad: i32,
    pub offset: i32,
    pub qn: i32,
    pub drafts: DevPtr,
    pub vq: DevPtr,
    pub qstat: DevPtr,
    pub sp: DevPtr,
    pub recv: [DevPtr; 8],
    pub tp: i32,
    pub rank: i32,
    pub seq: DevPtr,
    pub rowres: DevPtr,
    pub out: DevPtr,
    pub next: DevPtr,
    pub force_fb: i32,
    pub last_only: i32,
    pub spart: DevPtr,
    pub nquads: i32,
    pub so_s2: i32,
    pub so_fr: i32,
    pub leps: f32,
    pub so_bv: i32,
    pub block: i32,
}

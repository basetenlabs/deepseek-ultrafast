use anyhow::{Context, Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, DevPtr};
use deepseek_ultrafast_kernels::Kernels;
use deepseek_ultrafast_kv_transfer::Encoding;

pub fn pack(
    kernels: &Kernels,
    stream: &cu::Stream,
    encoding: Encoding,
    src: DevPtr,
    dst: DevPtr,
    valid: DevPtr,
    elements: usize,
) -> Result<()> {
    let group = encoding.group().context("cannot FP4-pack a raw range")?;
    ensure!(elements.is_multiple_of(group), "unaligned FP4 input");
    if elements == 0 {
        return Ok(());
    }
    let grid = u32::try_from(elements.div_ceil(256))?;
    let mut args = cu::Args::new();
    args.ptr(src)
        .ptr(dst)
        .ptr(dst + (elements / 2) as u64)
        .ptr(valid)
        .push(elements);
    cu::launch(
        kernels.get(&format!("kv_fp4_pack{group}"))?,
        &cu::Launch::new(grid, 256),
        stream,
        &mut args,
    )
}

pub fn unpack(
    kernels: &Kernels,
    stream: &cu::Stream,
    encoding: Encoding,
    src: DevPtr,
    dst: DevPtr,
    elements: usize,
) -> Result<()> {
    let group = encoding.group().context("cannot FP4-unpack a raw range")?;
    ensure!(elements.is_multiple_of(group), "unaligned FP4 output");
    if elements == 0 {
        return Ok(());
    }
    let grid = u32::try_from(elements.div_ceil(256))?;
    let mut args = cu::Args::new();
    args.ptr(src)
        .ptr(src + (elements / 2) as u64)
        .ptr(dst)
        .push(elements);
    cu::launch(
        kernels.get(&format!("kv_fp4_unpack{group}"))?,
        &cu::Launch::new(grid, 256),
        stream,
        &mut args,
    )
}

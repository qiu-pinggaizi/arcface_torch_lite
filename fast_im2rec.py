#!/usr/bin/env python3
"""
fast_im2rec.py - 完整的 mxnet im2rec 替代品（lst + rec 生成）

功能:
1. --list: 生成 lst 文件（单遍 os.walk + 流式写入）
2. 默认: 从 lst 生成 rec 文件（多进程加速）

用法:
    # 步骤1: 生成 lst
    python fast_im2rec.py --list --recursive train /data/WebFace42M

    # 步骤2: 生成 rec
    python fast_im2rec.py --num-thread 16 --quality 100 train /data/WebFace42M
"""

import os
import sys
import argparse
import random
import tempfile
import shutil
import time
import traceback


# ============================================================================
# 通用工具
# ============================================================================

class ProgressTracker:
    """简单的进度追踪器（无外部依赖）"""
    def __init__(self, total=None, desc='Processing'):
        self.total = total
        self.desc = desc
        self.count = 0
        self._last_pct = -5

    def update(self, n=1):
        self.count += n
        if self.total:
            pct = 100 * self.count / self.total
            if pct - self._last_pct >= 5:
                print(f'\r{self.desc}: {self.count}/{self.total} ({pct:.1f}%)',
                      end='', flush=True)
                self._last_pct = pct
        elif self.count % 10000 == 0:
            print(f'\r{self.desc}: {self.count}', end='', flush=True)

    def close(self):
        if self.total:
            print(f'\r{self.desc}: {self.count}/{self.total} (100%)')
        else:
            print(f'\r{self.desc}: {self.count} done')


# ============================================================================
# 第一部分: LST 生成（--list 模式）
# ============================================================================

def list_image_recursive(root, exts_set):
    """
    单遍 os.walk 遍历目录树，直接用 os.walk 返回的 filenames 过滤扩展名。
    无双遍历、无多线程（不需要，os.walk 内部已用 scandir，瓶颈在磁盘 I/O）。

    Yields: (index, relpath, label)
    """
    cat = {}
    idx = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        # 纯字符串过滤，无额外 I/O
        image_files = sorted(
            fname for fname in filenames
            if os.path.splitext(fname)[1].lower() in exts_set
        )
        if not image_files:
            continue

        if dirpath not in cat:
            cat[dirpath] = len(cat)
        label = cat[dirpath]

        for fname in image_files:
            yield (idx, os.path.relpath(os.path.join(dirpath, fname), root), label)
            idx += 1

    # 打印 label 对照表（与原版 im2rec 兼容）
    for k, v in sorted(cat.items(), key=lambda x: x[1]):
        print(os.path.relpath(k, root), v)


def list_image_flat(root, exts_set):
    """非递归模式：只扫描 root 目录下的文件"""
    idx = 0
    for fname in sorted(os.listdir(root)):
        fpath = os.path.join(root, fname)
        suffix = os.path.splitext(fname)[1].lower()
        if os.path.isfile(fpath) and suffix in exts_set:
            yield (idx, fname, 0)
            idx += 1


def shuffle_lst_file(input_path, output_path, seed=100, memory_limit_mb=2048):
    """
    智能 shuffle lst 文件。
    小文件: 全量内存 shuffle（精确）
    大文件: 分块 shuffle（近似但省内存）
    """
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)

    if file_size_mb < memory_limit_mb:
        print(f'Shuffling {file_size_mb:.1f}MB (in-memory)...')
        with open(input_path) as f:
            lines = f.readlines()
        random.seed(seed)
        random.shuffle(lines)
        with open(output_path, 'w') as f:
            f.writelines(lines)
    else:
        print(f'Shuffling {file_size_mb:.1f}MB (chunked, ~{memory_limit_mb}MB per chunk)...')
        chunk_line_count = int(memory_limit_mb * 1024 * 1024 / 100)  # ~100 bytes/line
        random.seed(seed)
        temp_dir = tempfile.mkdtemp(prefix='lst_shuffle_')
        try:
            chunk_files = []
            with open(input_path) as f:
                chunk_id = 0
                while True:
                    lines = []
                    for _ in range(chunk_line_count):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    if not lines:
                        break
                    random.shuffle(lines)
                    chunk_path = os.path.join(temp_dir, f'chunk_{chunk_id}.lst')
                    with open(chunk_path, 'w') as fc:
                        fc.writelines(lines)
                    chunk_files.append(chunk_path)
                    chunk_id += 1

            random.shuffle(chunk_files)
            with open(output_path, 'w') as fout:
                for cf in chunk_files:
                    with open(cf) as f:
                        shutil.copyfileobj(f, fout)
            print(f'Shuffled {chunk_id} chunks')
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def make_list(args):
    """生成 lst 文件（--list 模式入口）"""
    root = os.path.abspath(args.root)
    prefix = os.path.abspath(args.prefix)
    exts_set = set(args.exts)

    print('=== LST Generation Mode ===')
    print(f'Root: {root}')
    print(f'Recursive: {args.recursive}')
    print(f'Shuffle: {args.shuffle}\n')
    t0 = time.time()

    temp_lst = prefix + '.tmp.lst'
    try:
        # 流式写入 lst（内存 ~0）
        n_images = 0
        gen = list_image_recursive(root, exts_set) if args.recursive \
              else list_image_flat(root, exts_set)

        with open(temp_lst, 'w', buffering=8 * 1024 * 1024) as fout:
            for idx, relpath, label in gen:
                fout.write(f'{idx}\t{label}\t{relpath}\n')
                n_images += 1
                if n_images % 100000 == 0:
                    print(f'\rScanning: {n_images} images...', end='', flush=True)

        elapsed = time.time() - t0
        print(f'\rFound {n_images} images in {elapsed:.1f}s '
              f'({n_images/max(elapsed,0.1):.0f} images/s)')

        if n_images == 0:
            print('Error: No images found')
            os.remove(temp_lst)
            return 1

        # Shuffle（如果需要）
        if args.shuffle:
            shuffled = prefix + '.shuffled.tmp.lst'
            shuffle_lst_file(temp_lst, shuffled, seed=100)
            os.remove(temp_lst)
            temp_lst = shuffled

        # 重命名为最终文件
        final_path = prefix + '.lst'
        os.rename(temp_lst, final_path)
        temp_lst = None

        print(f'\nWrote {final_path}')
        print(f'Total: {time.time()-t0:.1f}s')
        return 0

    except KeyboardInterrupt:
        print('\nInterrupted')
        return 130
    finally:
        if temp_lst and os.path.exists(temp_lst):
            os.remove(temp_lst)


# ============================================================================
# 第二部分: REC 生成（非 --list 模式）
# ============================================================================

def read_list(path_in):
    """读取 lst 文件（生成器，不一次性加载）"""
    with open(path_in) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = [i.strip() for i in line.split('\t')]
            if len(parts) < 3:
                continue
            try:
                item = [int(parts[0])] + [parts[-1]] + [float(i) for i in parts[1:-1]]
                yield item
            except Exception:
                continue


def image_encode(encode_args, i, item, root):
    """
    编码单张图片为 MXNet RecordIO 格式。
    encode_args: dict，从 argparse.Namespace 提取的纯 dict（可 pickle）。
    返回: (index, packed_bytes, item) 或 (index, None, item)
    """
    import mxnet as mx
    import cv2

    fullpath = os.path.join(root, item[1])

    # Header
    if len(item) > 3 and encode_args['pack_label']:
        header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

    # Pass-through mode
    if encode_args['pass_through']:
        try:
            with open(fullpath, 'rb') as fin:
                img = fin.read()
            return (i, mx.recordio.pack(header, img), item)
        except Exception:
            return (i, None, item)

    # 读取图片
    try:
        img = cv2.imread(fullpath, encode_args['color'])
    except Exception:
        return (i, None, item)

    if img is None:
        return (i, None, item)

    # Center crop
    if encode_args['center_crop']:
        if img.shape[0] > img.shape[1]:
            margin = (img.shape[0] - img.shape[1]) // 2
            img = img[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) // 2
            img = img[:, margin:margin + img.shape[0]]

    # Resize
    if encode_args['resize']:
        if img.shape[0] > img.shape[1]:
            newsize = (encode_args['resize'],
                       img.shape[0] * encode_args['resize'] // img.shape[1])
        else:
            newsize = (img.shape[1] * encode_args['resize'] // img.shape[0],
                       encode_args['resize'])
        img = cv2.resize(img, newsize)

    # Pack
    try:
        s = mx.recordio.pack_img(header, img,
                                 quality=encode_args['quality'],
                                 img_fmt=encode_args['encoding'])
        return (i, s, item)
    except Exception:
        return (i, None, item)


def process_batch(encode_args, batch, root):
    """处理一批图片（用于多进程 worker）"""
    results = []
    for i, item in batch:
        results.append(image_encode(encode_args, i, item, root))
    return results


def make_record(args):
    """生成 rec 文件（非 --list 模式入口）"""
    import mxnet as mx
    from concurrent.futures import ProcessPoolExecutor, as_completed

    root = os.path.abspath(args.root)
    prefix = os.path.abspath(args.prefix)

    # 提取编码参数为纯 dict（可 pickle，传给子进程）
    encode_args = {
        'pack_label': args.pack_label,
        'pass_through': args.pass_through,
        'color': args.color,
        'center_crop': args.center_crop,
        'resize': args.resize,
        'quality': args.quality,
        'encoding': args.encoding,
    }

    # 查找 lst 文件
    if os.path.isdir(prefix):
        working_dir = prefix
    else:
        working_dir = os.path.dirname(prefix)

    prefix_base = os.path.basename(prefix)
    lst_files = sorted(
        os.path.join(working_dir, f) for f in os.listdir(working_dir)
        if f.startswith(prefix_base) and f.endswith('.lst')
    )

    if not lst_files:
        print(f'Error: No .lst files found with prefix {prefix}')
        return 1

    print(f'=== REC Generation Mode ===')
    print(f'Found {len(lst_files)} lst file(s)')
    print(f'Threads: {args.num_thread}')
    print(f'Quality: {args.quality}\n')

    for lst_file in lst_files:
        print(f'Processing {lst_file}...')
        t0 = time.time()

        # 输出文件
        base_name = os.path.splitext(os.path.basename(lst_file))[0]
        rec_path = os.path.join(working_dir, base_name + '.rec')
        idx_path = os.path.join(working_dir, base_name + '.idx')
        record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

        encoded = 0
        failed = 0

        if args.num_thread > 1:
            # ---- 多进程模式: 迭代器 + bounded submit ----
            print(f'  Using {args.num_thread} processes (batch=1000)...')

            BATCH_SIZE = 1000
            MAX_INFLIGHT = args.num_thread * 3  # bounded: 最多 N*3 个 batch 在飞

            executor = ProcessPoolExecutor(max_workers=args.num_thread)
            futures = {}
            batch = []
            batch_idx = 0

            for item in read_list(lst_file):
                batch.append((batch_idx, item))
                batch_idx += 1

                if len(batch) >= BATCH_SIZE:
                    # bounded submit: 如果在飞的 future 太多，先消费一些
                    if len(futures) >= MAX_INFLIGHT:
                        done_futures = []
                        for fut in list(futures):
                            if fut.done():
                                done_futures.append(fut)
                        if not done_futures:
                            # 没有完成的，等一个
                            done_fut = next(iter(futures))
                            done_futures = [done_fut]
                            done_fut.result()  # 阻塞等完成
                        for fut in done_futures:
                            for _, s, it in fut.result():
                                if s is not None:
                                    record.write_idx(it[0], s)
                                    encoded += 1
                                else:
                                    failed += 1
                            del futures[fut]
                        if (encoded + failed) % 50000 < BATCH_SIZE:
                            elapsed = time.time() - t0
                            speed = (encoded + failed) / max(elapsed, 0.1)
                            print(f'\r  Progress: {encoded+failed} '
                                  f'(ok={encoded}, fail={failed}, '
                                  f'{speed:.0f} imgs/s)', end='', flush=True)

                    futures[executor.submit(
                        process_batch, encode_args, batch, root)] = True
                    batch = []

            # 提交剩余 batch
            if batch:
                futures[executor.submit(
                    process_batch, encode_args, batch, root)] = True

            # 消费剩余 futures
            for fut in as_completed(futures):
                for _, s, it in fut.result():
                    if s is not None:
                        record.write_idx(it[0], s)
                        encoded += 1
                    else:
                        failed += 1

            executor.shutdown(wait=False)
            print()  # 换行

        else:
            # ---- 单线程模式 ----
            print(f'  Single-threaded mode...')

            for i, item in enumerate(read_list(lst_file)):
                result = image_encode(encode_args, i, item, root)
                _, s, it = result
                if s is not None:
                    record.write_idx(it[0], s)
                    encoded += 1
                else:
                    failed += 1
                if (i + 1) % 10000 == 0:
                    elapsed = time.time() - t0
                    speed = (i + 1) / max(elapsed, 0.1)
                    print(f'\r  Progress: {i+1} ({speed:.0f} imgs/s)',
                          end='', flush=True)

            print()

        record.close()

        elapsed = time.time() - t0
        print(f'  Wrote {rec_path}')
        print(f'  Success: {encoded}, Failed: {failed}')
        print(f'  Time: {elapsed:.1f}s ({encoded/max(elapsed,0.1):.0f} imgs/s)\n')

    print('All done!')
    return 0


# ============================================================================
# 命令行参数（兼容原版 mxnet im2rec 参数名）
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fast im2rec: mxnet im2rec replacement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('prefix', help='prefix of input/output lst and rec files')
    parser.add_argument('root', help='path to folder containing images')

    # LST 生成选项
    cgroup = parser.add_argument_group('Options for creating image lists (--list mode)')
    cgroup.add_argument('--list', action='store_true',
                        help='Create image list by traversing root folder')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png'],
                        help='acceptable image extensions')
    cgroup.add_argument('--recursive', action='store_true', default=False,
                        help='recursively walk through subdirs')
    cgroup.add_argument('--shuffle', action='store_true', default=True,
                        help='randomize image order')
    cgroup.add_argument('--no-shuffle', dest='shuffle', action='store_false')

    # REC 生成选项
    rgroup = parser.add_argument_group('Options for creating database (default mode)')
    rgroup.add_argument('--pass-through', action='store_true',
                        help='save image as-is without transformation')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize shorter edge to this size')
    rgroup.add_argument('--center-crop', action='store_true',
                        help='crop center to make it square')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality (1-100) or PNG compression (1-9)')
    rgroup.add_argument('--num-thread', '--threads', type=int, default=1,
                        help='number of threads/processes for encoding')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='color mode: 1=color, 0=grayscale, -1=with alpha')
    rgroup.add_argument('--encoding', type=str, default='.jpg',
                        choices=['.jpg', '.png'],
                        help='image encoding format')
    rgroup.add_argument('--pack-label', action='store_true',
                        help='pack multi-dimensional label')

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if args.list:
            return make_list(args)
        else:
            return make_record(args)
    except KeyboardInterrupt:
        print('\nInterrupted')
        return 130
    except Exception as e:
        print(f'\nError: {e}')
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

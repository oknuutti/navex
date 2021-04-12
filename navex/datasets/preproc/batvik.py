import re
import os
import argparse
import logging
import warnings

from tqdm import tqdm
import numpy as np
import cv2

from navex.datasets.tools import find_files_recurse


def main():
    """
    Extract frames from Nokia Båtvik videos
    """
    parser = argparse.ArgumentParser('Extract frames from Nokia Båtvik videos')

    parser.add_argument('--src', help="input folder")
    parser.add_argument('--dst', help="output folder")
    parser.add_argument('--index', default='dataset_all.txt', help="index file in the output folder")
    parser.add_argument('--subset', help='process only this subfolder')
    parser.add_argument('--regex', default=r'(^|_|-|/|\\)HD_CAM(_|-)', help="regex for video file names")
    parser.add_argument('--start', type=float, default=0.0, help="where to start processing [0-1]")
    parser.add_argument('--end', type=float, default=1.0, help="where to stop processing [0-1]")
    parser.add_argument('--skip-first', type=int, default=100, help="discard this many frames from beginning")
    parser.add_argument('--skip-last', type=int, default=100, help="discard this many frames from the end")
    parser.add_argument('--scale', type=float, default=1.0, help="scale images by this much")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", message='NAL unit')   # ignoring doesn't work, regrettably
    do_scaling = not np.isclose(args.scale, 1.0)

    os.makedirs(args.dst, exist_ok=True)
    index_file = os.path.join(args.dst, args.index)
    index_rows = []
    processed = {}

    def write_index(entries=tuple(), head=False):
        with open(index_file, 'w' if head else 'a') as fh:
            if head:
                fh.write('id\trand\tpresent\tfile\n')
            for entry in entries:
                fh.write('%d\t%f\t%d\t%s\n' % tuple(entry))

    if not os.path.exists(index_file):
        write_index(head=True)
    else:
        with open(index_file, 'r') as fh:
            for i, row in enumerate(fh):
                if i == 0:
                    continue
                id, rand, pres, file = row.strip().split('\t')
                entry = [int(id), float(rand), int(pres), file]
                processed[file] = entry
                index_rows.append(entry)
    videos = find_files_recurse(args.src, ext='mp4')

    # `id` is a global image id
    id, v_ids = 0, {}

    for video in videos:
        if re.search(args.regex, video):
            logging.info('processing file %s...' % video)

            subset = os.path.dirname(video).split(os.sep)[-1]
            if args.subset and args.subset != subset:
                continue
            os.makedirs(os.path.join(args.dst, subset), exist_ok=True)

            # `i` is an id for each video file in each subfolder
            video_name = os.path.basename(video).split(os.sep)[-1]
            if subset not in v_ids:
                v_ids[subset] = {'__next__': 1}
            i = v_ids[subset][video_name] = v_ids[subset]['__next__']
            v_ids[subset]['__next__'] += 1

            cap = cv2.VideoCapture(video)
            n, j, add_count = cap.get(cv2.CAP_PROP_FRAME_COUNT), 0, 0       # `j` is frame number in each video
            pbar = tqdm(total=n, mininterval=3)

            while cap.isOpened():
                ok = True
                j += 1
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    logging.error(str(e))
                    ok = False

                file = os.path.join(subset, 'vid-%d-frame-%d.png' % (i, j))
                if file in processed and (processed[file][2] or not (args.start < processed[file][1] < args.end)):
                    continue
                if j < args.skip_first:
                    continue
                elif n - j < args.skip_last:
                    break

                if ok and frame is not None and len(frame) > 0:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ok = test_img(img)
                else:
                    ok = False

                if file in processed:
                    rand = processed[file][1] if ok else -1
                    processed[file][1] = rand
                else:
                    rand = np.random.uniform(0, 1) if ok else -1

                added = 0
                if args.start < rand < args.end:
                    if do_scaling:
                        img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
                    cv2.imwrite(os.path.join(args.dst, file), img, (cv2.IMWRITE_PNG_COMPRESSION, 9))
                    added = 1

                id += 1
                add_count += added
                if file in processed:
                    processed[file][2] = added or processed[file][2]
                elif len(processed) > 0:
                    # reached end of previously written record, dump reprocessed entries
                    write_index(index_rows, head=True)
                    processed, index_rows = {}, []
                else:
                    write_index([(id, rand, added, file)])

                pbar.update(1)
                pbar.set_postfix({'added': add_count, 'ratio': add_count/j}, refresh=False)

            cap.release()

    if len(processed) > 0:
        write_index(index_rows, head=True)


def test_img(img):
    return True


if __name__ == '__main__':
    main()

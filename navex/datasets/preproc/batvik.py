import copy
import re
import os
import sys
import math
import argparse
import logging
import shutil
import warnings

from tqdm import tqdm
import numpy as np
import cv2

try:
    from kapture.io.csv import kapture_from_dir
    from kapture.io.records import get_record_fullpath
except ModuleNotFoundError:
    warnings.warn("Kapture not available, cannot preprocess geo-referenced data")

from .tools import create_image_pairs, safe_split
from ..tools import find_files_recurse, ImageDB, angle_between_v, rotate_expand_border, q_times_v, vector_rejection, \
    Camera, unit_aflow, save_aflow


def main():
    """
    Construct image pairs from geo-referenced Nokia Båtvik data
     - for georeferencing, see hw_visnav.preprocess and .depthmaps
    """
    parser = argparse.ArgumentParser('Construct image pairs from geo-referenced Nokia Båtvik data')

    parser.add_argument('--type', default='georef', choices=('georef', 'simple', 'sat'),
                        help="Type of preprocessing to do")
    parser.add_argument('--src', action='append', required=True, help="source folders")
    parser.add_argument('--dst', required=True, help="output folder")
    parser.add_argument('--index', default='dataset_all.sqlite',
                        help="index file name in the output folder")
    parser.add_argument('--start', type=float, default=0.0, help="where to start processing [0-1]")
    parser.add_argument('--end', type=float, default=1.0, help="where to stop processing [0-1]")

    parser.add_argument('--pairs', default='pairs.txt', help="pairing file to create in root")
    parser.add_argument('--aflow', default='aflow', help="subfolder where the aflow files are generated")
    parser.add_argument('--img-max', type=int, default=3, help="how many times same images can be repeated in pairs")
    parser.add_argument('--max-angle', type=float, default=0,
                        help="max angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--min-matches', type=int, default=10000,
                        help="min pixel matches in order to approve generated pair")

    parser.add_argument('--aflow-match-coef', type=float, default=1.0,
                        help="expand pixel matching distance by setting value larger than 1.0")
    parser.add_argument('--trust-georef', action='store_true', help='Trust georeferencing; don\'t estimate aflow using ICP')

    parser.add_argument('--overwrite', action='store_true', help='clear all, overwrite rotated images')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    SENSOR_NAME = 'cam'

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.dst, args.index)
    source_paths = {src_path.split(os.sep)[-1]: src_path for src_path in args.src}

    def pathfun(fname, subfolder, ext):
        subset, *rest = fname.split(os.sep)
        return os.path.join(source_paths[subset], 'exr', subfolder, *rest)[:-4] + ext

    geom_pathfun = lambda fname: pathfun(fname, 'geometry', '.xyz.exr')
    depth_pathfun = lambda fname: pathfun(fname, 'depth', '.d.exr')

    if args.overwrite or not os.path.exists(index_path):
        logging.info('Building the index by scanning the source folders...')
        index = ImageDB(index_path, truncate=True)

        frames, angles = [], {}
        for src_path in args.src:
            subset_path = src_path.split(os.sep)[-1]
            kapt_path = os.path.join(src_path, 'kapture')
            kapt = kapture_from_dir(kapt_path)
            sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_p = get_cam_params(kapt, SENSOR_NAME)
            hz_fov = math.degrees(2 * math.atan(width/2/fl_x))
            set_id = index.set_subset(subset_path, *cam_p[1:12])

            for fid, img_files in tqdm(kapt.records_camera.items(), desc='Copying images from %s' % src_path):
                src_path = get_record_fullpath(kapt_path, img_files[sensor_id])
                rel_dst_path = os.path.join(subset_path, src_path.split('/')[-1])

                geom_path = geom_pathfun(rel_dst_path)
                if not os.path.exists(geom_path):
                    # dont include image if geometry data missing
                    continue

                dst_path = os.path.join(args.dst, rel_dst_path)

                ori = kapt.trajectories[fid][sensor_id].r
                loc = kapt.trajectories[fid][sensor_id].t.flatten()
                angle = rotate_image(src_path, dst_path, ori)
                angles[rel_dst_path] = angle

                frames.append((set_id, rel_dst_path, hz_fov, angle) + safe_split(ori, True) + safe_split(loc, False)
                              + (1, 0, 0, 0))

        frames = sorted(frames, key=lambda x: x[0])
        index.add(('id', 'set_id', 'file', 'hz_fov', 'img_angle',
                   'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz',
                   'sc_trg_x', 'sc_trg_y', 'sc_trg_z',
                   'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz'),
                  [(i, *frame) for i, frame in enumerate(frames)])
    else:
        index = ImageDB(index_path)

    create_image_pairs(args.dst, index, args.pairs, geom_pathfun, args.aflow, args.img_max, None,
                       0, args.max_angle, args.min_matches, read_meta=True, start=args.start,
                       end=args.end, exclude_shadowed=False, across_subsets=True, trust_georef=args.trust_georef,
                       ignore_img_angle=False, cluster_unit_vects=False, depth_src=depth_pathfun,
                       aflow_match_coef=args.aflow_match_coef, depth_is_along_zaxis=True)


def rotate_image(src, dst, ori):
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    assert img is not None, 'image not found at %s' % src

    # assume in cam frame: +z down (cam axis), -y towards north (up), +x is right wing (east)
    north_v = np.array([0, -1, 0])
    cam_axis = np.array([0, 0, 1])
    cam_up = np.array([0, -1, 0])

    sc_north = q_times_v(ori, north_v)
    img_north = vector_rejection(sc_north, cam_axis)
    angle = angle_between_v(cam_up, img_north, direction=cam_axis)

    rimg = rotate_expand_border(img, angle, fullsize=True, lib='opencv')

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, rimg, (cv2.IMWRITE_PNG_COMPRESSION, 9))

    return angle


def get_cam_params(kapt, sensor_name):
    sid, sensor = None, None
    for id, s in kapt.sensors.items():
        if s.name == sensor_name:
            sid, sensor = id, s
            break
    sp = sensor.sensor_params
    return (sid,) + tuple(map(int, sp[1:3])) + tuple(map(float, sp[3:]))


def main_sat_imgs():
    """
    Crate image pairs from satellite images of same scene.
    """
    parser = argparse.ArgumentParser('Extract frames from satellite images')

    parser.add_argument('--type', default='georef', choices=('georef', 'simple', 'sat'),
                        help="Type of preprocessing to do")
    parser.add_argument('--src', help="input folder")
    parser.add_argument('--dst', help="output folder")
    parser.add_argument('--index', default='dataset_all.sqlite', help="index file in the output folder")
    parser.add_argument('--regex', help="regex for subfolder names")
    parser.add_argument('--start', type=float, default=0.0, help="where to start processing [0-1]")
    parser.add_argument('--end', type=float, default=1.0, help="where to stop processing [0-1]")
    parser.add_argument('--log', default='INFO', help="log level")
    parser.add_argument('--auto-season', action='store_true',
                        help='determine season from date if not specified using *_(w|t|s).(jpg|png) naming convention')
    parser.add_argument('--overwrite', action='store_true', help='clear all, overwrite rotated images')
    args = parser.parse_args()

    logging.basicConfig(level=args.log)

    os.makedirs(os.path.join(args.dst, 'aflow'), exist_ok=True)
    index_path = os.path.join(args.dst, args.index)

    if args.overwrite or not os.path.exists(index_path):
        logging.info('Building the index by scanning the source folders...')
        index = ImageDB(index_path, truncate=True)

        frames = []
        scene_id = 0
        for scene_path in tqdm(os.listdir(args.src), desc='Scanning source folders'):
            if args.regex and not re.match(args.regex, scene_path) \
                    or not os.path.isdir(os.path.join(args.src, scene_path)):
                continue

            scene_id += 1
            os.makedirs(os.path.join(args.dst, scene_path), exist_ok=True)

            for img_path in os.listdir(os.path.join(args.src, scene_path)):
                if img_path[-4:] not in ('.jpg', '.png'):
                    continue

                season = img_path.split('_')[-1][0]     # w: winter, t: transition (spring/autumn), s: summer
                if season not in ('w', 't', 's'):
                    if args.auto_season:
                        month = int(img_path.split('_')[1][:2])
                        season = 'w' if month in (12, 1, 2, 3) else 's' if month in (6, 7, 8, 9) else 't'
                    else:
                        season = 't'
                set_id = scene_id * 4 + {'w': 0, 't': 1, 's': 2}[season]

                src_path = os.path.join(args.src, scene_path, img_path)
                dst_path = os.path.join(args.dst, scene_path, img_path)
                shutil.copyfile(src_path, dst_path)

                frames.append((set_id, '/'.join((scene_path, img_path))))

        frames = sorted(frames, key=lambda x: x[0])
        index.add(('id', 'set_id', 'file'), [(i, *frame) for i, frame in enumerate(frames)])
    else:
        index = ImageDB(index_path)

    pairs = []
    pairs_path = os.path.join(args.dst, 'pairs.txt')
    if args.overwrite or not os.path.exists(pairs_path):
        # create pairs
        scenes = {}
        for id, set_id, fname in index.get_all(('id', 'set_id', 'file'), start=args.start, end=args.end):
            scene_id, season_id = divmod(set_id, 4)
            scenes.setdefault(scene_id, {}).setdefault(season_id == 0, []).append([id, fname, None, None])

        for scene_id, scene_imgs in tqdm(scenes.items(), desc='Creating pairs'):
            scene_imgs.setdefault(True, {})

            shapes, ref_img, ref_offset = [], None, [0, 0]
            for temp in scene_imgs.values():
                for frame_info in temp:
                    id1, fname, _, _ = frame_info
                    frame_info[2] = cv2.imread(os.path.join(args.dst, fname), cv2.IMREAD_UNCHANGED)
                    if ref_img is None:
                        ref_img, offset = frame_info[2], [0, 0]
                    else:
                        offset = get_offset(ref_img, frame_info[2])
                        ref_offset[0] = max(ref_offset[0], -offset[0])
                        ref_offset[1] = max(ref_offset[1], -offset[1])
                    frame_info[3] = offset
                    shapes.append([*frame_info[2].shape[:2], *offset])

            # common width and height
            shapes = np.array(shapes)
            h, w = np.min(shapes[:, :2] - (shapes[:, 2:] + ref_offset), axis=0)

            def crop_img(fname, img, offset):
                ox, oy = np.add(offset, ref_offset)
                if img.shape[:2] != (h, w) or offset[0] != 0 or offset[1] != 0:
                    cv2.imwrite(os.path.join(args.dst, fname), img[oy:oy + h, ox:ox + w],
                                (cv2.IMWRITE_JPEG_QUALITY, 99) if fname.endswith('.jpg')
                                else (cv2.IMWRITE_PNG_COMPRESSION, 9))

            for id1, fname1, img1, offset1 in scene_imgs[True]:
                crop_img(fname1, img1, offset1)

                for id2, fname2, img2, offset2 in scene_imgs[False]:
                    crop_img(fname2, img2, offset2)
                    pairs.append((id1, id2, fname1, fname2, w, h))

        with open(pairs_path, 'w') as fh:
            fh.write('id1 id2 file1 file2 w h\n')
            for id_i, id_j, fname_i, fname_j, w, h in pairs:
                fh.write('%d %d %s %s %d %d\n' % (id_i, id_j, fname_i, fname_j, w, h))

        logging.info('Wrote %d pairs to %s', len(pairs), pairs_path)

    else:
        logging.info('Reading the existing pair file...')
        pairs = []
        with open(pairs_path, 'r') as fh:
            for k, line in enumerate(fh):
                if k == 0:
                    continue
                id1, id2, file1, file2, w, h = line.split(' ')
                pairs.append((id1, id2, file1, file2, w, h))

    # write unit aflow files
    for id1, id2, fname1, fname2, w, h in tqdm(pairs, desc='Writing unit aflow files'):
        aflow_path = os.path.join(args.dst, 'aflow', '%d_%d.png' % (id1, id2))
        aflow = unit_aflow(w, h)
        if args.overwrite or not os.path.exists(aflow_path):
            save_aflow(aflow_path, aflow)


def get_offset(ref_img, img):
    """
    Find offset between images using phase correlation, assumes that the image scales are the same
    """

    # TODO

    return [0, 0]


def main_simple():
    """
    Extract frames from Nokia Båtvik videos
    """
    parser = argparse.ArgumentParser('Extract frames from Nokia Båtvik videos')

    parser.add_argument('--type', default='georef', choices=('georef', 'simple', 'sat'),
                        help="Type of preprocessing to do")
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
    if np.any([re.match(r'(-{1,2}type=)?simple', a) for a in sys.argv]):
        main_simple()
    elif np.any([re.match(r'(-{1,2}type=)?sat', a) for a in sys.argv]):
        main_sat_imgs()
    else:
        main()

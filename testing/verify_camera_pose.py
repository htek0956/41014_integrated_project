import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from robot import Robot
from path_planning import map_from_scene, PlannedLoopReference, DEFAULT_SCENE, plot_plan


def verify(headless=True, drive=False):
    if drive:
        import main
        main.LOG_RESULTS = True
        data = main.main(show_viewer=False, camera_diagnostics=True)
        samples = data['camera_measurements']
        accepted = [item for item in samples if item['pose'] is not None]
        errors = np.array([item['pose']-item['truth'] for item in accepted])
        errors[:, 2] = (errors[:, 2]+np.pi) % (2*np.pi)-np.pi
        metrics = dict(mode='actual simulated motion, including docking',
            samples=len(samples), valid_samples=len(accepted), valid_percent=100*len(accepted)/len(samples),
            position_rms_m=float(np.sqrt(np.mean(np.sum(errors[:, :2]**2, axis=1)))),
            position_max_m=float(np.linalg.norm(errors[:, :2], axis=1).max()),
            heading_rms_deg=float(np.rad2deg(np.sqrt(np.mean(errors[:, 2]**2)))),
            multi_tag_samples=sum(len(item['ids_used'])>1 for item in accepted),
            docked=data['docked'])
        folder = ROOT / 'figures/camera_localisation'
        (folder / 'driving_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        grid, start = map_from_scene()
        fig = plot_plan(PlannedLoopReference(grid, start))
        truth = np.array([item['truth'] for item in samples])
        camera = np.array([item['pose'] for item in accepted])
        fig.axes[0].plot(truth[:, 0], truth[:, 1], '--', color='tab:orange', label='Driven truth (evaluation only)')
        fig.axes[0].scatter(camera[:, 0], camera[:, 1], s=8, color='tab:green', label='Camera-only measurements')
        fig.axes[0].legend()
        fig.savefig(folder / 'driving_pose_comparison.png', dpi=150, bbox_inches='tight')
        print(json.dumps(metrics, indent=2))
        assert metrics['docked'] and metrics['valid_percent'] > 90
        assert metrics['position_rms_m'] < .10 and metrics['heading_rms_deg'] < 5
        if not headless:
            plt.show()
        plt.close('all')
        return
    grid, start = map_from_scene()
    path = PlannedLoopReference(grid, start)
    robot = Robot(DEFAULT_SCENE, max_yaw_rate=2, show_viewer=False)
    parent = ROOT / 'figures' / 'camera_localisation'
    parent.mkdir(exist_ok=True)
    index = 1
    while (parent / f'run_{index:03d}').exists():
        index += 1
    folder = parent / f'run_{index:03d}'
    folder.mkdir()
    truth, estimated, counts, errors, rms, failures = [], [], [], [], [], []
    best_tag_count = 0
    used_all_count = 0
    try:
        for progress in np.arange(0, path.total_length, .25):
            reference = path.reference_at(progress)
            robot.reset((reference['x'], reference['y'], reference['heading']))


            frame = robot.read_camera()
            frame['timestamp'] = float(progress)
            measurement = robot.read_camera_pose(frame, annotate=True)
            actual = np.array(robot.get_truth())
            truth.append(actual)
            if not measurement['valid']:
                estimated.append([np.nan]*3)
                counts.append(0)
                failures.append(dict(progress_m=float(progress), reason=measurement['reason']))
                continue
            estimated.append(measurement['pose'])
            count = len(measurement['ids_used'])
            counts.append(count)
            error = measurement['pose']-actual
            error[2] = (error[2]+np.pi) % (2*np.pi)-np.pi
            errors.append(error)
            rms.append(measurement['reprojection_rms_px'])
            if len(measurement['ids_used']) == len(measurement['ids_detected']):
                used_all_count += 1
            if count > best_tag_count:
                best_tag_count = count
                cv2.imwrite(str(folder / 'detected_tags.png'), cv2.cvtColor(measurement['annotated_rgb'], cv2.COLOR_RGB2BGR))
            assert np.linalg.eigvalsh(measurement['covariance']).min() > 0

        blank = dict(timestamp=999., rgb=np.full((480, 640, 3), 127, dtype=np.uint8))
        missing = robot.read_camera_pose(blank)
        assert not missing['valid'] and missing['pose'] is None and missing['covariance'] is None
        assert missing['timestamp'] == 999.


        robot.reset((1.275, 0, -np.pi/2))
        fixed_image = robot.read_camera()
        first = robot.read_camera_pose(fixed_image)
        robot.reset((6, 4, .4))
        second = robot.read_camera_pose(fixed_image)
        assert first['valid'] and second['valid']
        np.testing.assert_allclose(first['pose'], second['pose'], atol=1e-10)
    finally:
        robot.close()
    actual, estimate, counts, errors = map(np.asarray, (truth, estimated, counts, errors))
    if not len(errors):
        raise AssertionError('No camera estimates were produced.')
    position_errors = np.linalg.norm(errors[:, :2], axis=1)
    metrics = dict(samples=len(actual), valid_samples=len(errors),
                   valid_percent=100*len(errors)/len(actual),
                   single_tag_samples=int(np.sum(counts == 1)),
                   multi_tag_samples=int(np.sum(counts > 1)), maximum_tags_used=int(counts.max()),
                   frames_using_every_detected_tag=used_all_count,
                   position_rms_m=float(np.sqrt(np.mean(position_errors**2))),
                   position_max_m=float(position_errors.max()),
                   heading_rms_deg=float(np.rad2deg(np.sqrt(np.mean(errors[:, 2]**2)))),
                   heading_max_deg=float(np.rad2deg(abs(errors[:, 2]).max())),
                   mean_reprojection_rms_px=float(np.mean(rms)), failures=failures,
                   evaluation='Ideal rendered images at route samples, not a noisy EKF consistency test',
                   covariance='Uncalibrated approximate joint-PnP covariance; validate before use as EKF R')
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    np.savez_compressed(folder / 'measurements.npz', truth=actual, camera_pose=estimate, tag_count=counts)
    fig = plot_plan(path)
    ax = fig.axes[0]
    ax.plot(actual[:, 0], actual[:, 1], color='tab:orange', label='Truth (verification only)')
    ax.scatter(estimate[:, 0], estimate[:, 1], s=12, color='tab:green', label='Camera-only pose')
    ax.legend()
    ax.set_title(f'Camera pose check: {metrics["valid_percent"]:.1f}% valid, position RMS {metrics["position_rms_m"]*100:.1f} cm')
    fig.savefig(folder / 'pose_comparison.png', dpi=150, bbox_inches='tight')
    fig2, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    valid = counts > 0
    indices = np.arange(len(actual))
    axes[0].plot(indices[valid], position_errors*100)
    axes[0].set_ylabel('Position error [cm]')
    axes[1].plot(indices[valid], np.rad2deg(errors[:, 2]))
    axes[1].set_ylabel('Heading error [deg]')
    axes[2].step(indices, counts)
    axes[2].set(ylabel='Tags used', xlabel='Route sample (0.25 m spacing)')
    for ax in axes:
        ax.grid(alpha=.3)
    fig2.tight_layout()
    fig2.savefig(folder / 'pose_errors.png', dpi=150)
    print(json.dumps(metrics, indent=2))
    print(f'Report: {folder}')
    assert metrics['multi_tag_samples'] > 0, 'No multi-tag estimates were exercised.'
    assert metrics['position_rms_m'] < .10 and metrics['heading_rms_deg'] < 5, 'Pose errors need investigation.'
    if not headless:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--drive', action='store_true', help='Verify during actual driving and docking')
    args = parser.parse_args()
    verify(args.headless, args.drive)

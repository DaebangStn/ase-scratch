from isaacgym.gymutil import parse_arguments
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.poselib.visualization.common import plot_skeleton_motion_interactive

params = [
    {"name": "--npy", "type": str, "default": False,
     "help": "Path to the npy file containing the skeleton motion."},
]


if __name__ == "__main__":
    args = parse_arguments(description="Skeleton Motion Player", custom_parameters=params)

    if not args.npy:
        raise ValueError("Please provide the path to the npy file containing the skeleton motion.")

    motion = SkeletonMotion.from_file(args.npy)
    plot_skeleton_motion_interactive(motion)

import os


def list_bag_files_auto(raw_dir):
    """List .bag files and skip sidecar compressed .c.bag files."""
    bag_files = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".bag"):
            continue
        if fname.endswith(".c.bag"):
            continue
        bag_files.append(
            {
                "link": "",
                "start": 0,
                "end": 1,
                "local_path": os.path.join(raw_dir, fname),
            }
        )
    return bag_files

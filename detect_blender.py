def detect_blender():
    """Detect existing Blender installation.

    Returns path to Blender executable or None if not found.
    Detection priority:
    1. BLENDER_PATH env var (explicit override)
    2. System PATH ('blender' command)
    3. Platform-specific common locations
    4. Bundled lib/blender/ directory
    """
    import shutil

    # 1. Check BLENDER_PATH env var (explicit override)
    blender_path = os.environ.get('BLENDER_PATH')
    if blender_path and os.path.isfile(blender_path):
        return blender_path

    # 2. Check system PATH
    blender_in_path = shutil.which('blender')
    if blender_in_path:
        return blender_in_path

    # 3. Check platform-specific common locations
    system = platform.system()
    common_locations = []

    if system == "Windows":
        # Windows: Check Program Files
        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        blender_foundation = Path(program_files) / "Blender Foundation"
        if blender_foundation.exists():
            # Find latest version (e.g., Blender 4.2, Blender 4.1, etc.)
            for blender_dir in sorted(blender_foundation.glob("Blender*"), reverse=True):
                exe = blender_dir / "blender.exe"
                if exe.exists():
                    common_locations.append(exe)

    elif system == "Darwin":  # macOS
        common_locations = [
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        ]

    else:  # Linux
        common_locations = [
            Path("/usr/bin/blender"),
            Path("/usr/local/bin/blender"),
            Path("/snap/bin/blender"),
        ]

    for loc in common_locations:
        if isinstance(loc, Path) and loc.exists():
            return str(loc)

    # 4. Check bundled lib/blender/
    node_root = Path(__file__).parent.absolute()
    blender_dir = node_root / "lib" / "blender"
    if blender_dir.exists():
        from installer.blender import find_blender_executable
        blender_exe = find_blender_executable(str(blender_dir))
        if blender_exe:
            return str(blender_exe)

    return None
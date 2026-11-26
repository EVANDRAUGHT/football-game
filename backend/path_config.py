import os


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_dir() -> str:
    data_dir = os.getenv("GOALSCOPE_DATA_DIR")
    if data_dir:
        return os.path.abspath(data_dir)
    return os.path.join(get_project_root(), "GoalScope_data")


def ensure_data_dirs() -> str:
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(get_upload_dir(), exist_ok=True)
    os.makedirs(get_export_dir(), exist_ok=True)
    os.makedirs(get_phase_a_cache_dir(), exist_ok=True)
    return data_dir


def get_upload_dir() -> str:
    return os.path.join(get_data_dir(), "uploads")


def get_export_dir() -> str:
    return os.path.join(get_upload_dir(), "exports")


def get_phase_a_cache_dir() -> str:
    return os.path.join(get_data_dir(), "phase_a_cache")


def get_database_path() -> str:
    return os.path.join(get_data_dir(), "football_demo.db")


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", f"sqlite:///{get_database_path()}")

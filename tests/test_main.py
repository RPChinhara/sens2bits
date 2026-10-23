from __future__ import annotations

from pathlib import Path


def test_package_main(monkeypatch):
    import sens2bits.main as package_main

    monkeypatch.setattr(package_main, "run_sensitivity", lambda: None)
    monkeypatch.setattr(package_main, "run_allocation", lambda: None)
    package_main.main()


def test_root_main(monkeypatch):
    root_main_path = Path("/home/rudrac/sens2bits/main.py")
    # avoid running heavy sensitivity/allocation work
    import sens2bits.main as package_main

    monkeypatch.setattr(package_main, "main", lambda: None)
    source = root_main_path.read_text()
    code = compile(source, str(root_main_path), "exec")
    namespace = {"__name__": "__main__", "__file__": str(root_main_path)}
    exec(code, namespace)

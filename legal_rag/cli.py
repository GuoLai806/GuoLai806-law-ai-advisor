"""Compatibility command for the legal profile."""
from evidence_rag.cli import main as _main

def main():
    return _main(default_profile='legal')

if __name__=='__main__': main()

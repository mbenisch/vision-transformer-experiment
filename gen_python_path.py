import os
import sys

def main():
    python_path = os.pathsep.join(sys.path)
    with open('.env', 'w') as f:
        f.write(f'PYTHONPATH={python_path}\n')

if __name__ == '__main__':
    main()
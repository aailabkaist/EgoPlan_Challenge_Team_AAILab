import sys
import os

def main():
    input = sys.argv[1:]
    sh_path = os.getcwd() + '/scripts/' +input[1] + '.sh'
    if input[0]=='pyname':
        try:
            with open(sh_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    stripped_line=line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        if '-u' in stripped_line:
                            parts = stripped_line.split()
                            for part in parts:
                                if part.endswith('.py'):
                                    pyname = part.split('.py')[0]
                                    print(pyname)
                                    return
        except FileNotFoundError:
            print(f"File not found: {sh_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    elif input[0]=='option':
        try:
            with open(sh_path, 'r') as file:
                lines = file.readlines()
                script_found = False
                for line in lines:
                    # Ignore comments and empty lines
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        # Check if the line contains 'python3 -u' and extract the script name and options
                        if 'python3 -u' in stripped_line:
                            script_found = True
                            continue
                        options = []
                        if script_found:
                            print(stripped_line)
                return
        except FileNotFoundError:
            print(f"File not found: {sh_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
if __name__ == "__main__":
    main()
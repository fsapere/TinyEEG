import sys
from pathlib import Path

ares_path = "/scratch/fsapere/ARES/ARES"
if ares_path not in sys.path:
    sys.path.append(ares_path)

from codegen.generate_c_code import CCodeGenerator

def main():
    target_hardware = "siracusa"
    artifacts_dir = Path("ares_artifacts")
    test_case_dir = artifacts_dir / "test_case_1"
    
    print(f"\n[3/3] Generating C Code for {target_hardware}...")
    generator = CCodeGenerator(
        target_name=target_hardware,
        network_info_path=artifacts_dir / "network_info.json",
        weights_dir=artifacts_dir / "weights",
        test_case_dir=test_case_dir,
        output_dir=artifacts_dir / "c_project"
    )
    generator.generate_all()
    
    print("\nGeneration completed! The C project is located in ares_artifacts/c_project")

if __name__ == "__main__":
    main()

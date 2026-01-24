#!/usr/bin/env python3
"""
Script to compare requirements.txt with installed package versions
"""

import pkg_resources
import re
from typing import Dict, List, Tuple

def parse_requirements_file(file_path: str) -> Dict[str, str]:
    """Parse requirements.txt file and extract package names and versions"""
    requirements = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle packages with extras like uvicorn[standard]
                if '[' in line:
                    package_name = line.split('[')[0]
                else:
                    package_name = line.split('==')[0] if '==' in line else line.split('>=')[0] if '>=' in line else line.split('<=')[0] if '<=' in line else line
                
                # Extract version
                if '==' in line:
                    version = line.split('==')[1]
                elif '>=' in line:
                    version = line.split('>=')[1]
                elif '<=' in line:
                    version = line.split('<=')[1]
                else:
                    version = "any"
                
                requirements[package_name.lower()] = version
    
    return requirements

def get_installed_versions() -> Dict[str, str]:
    """Get currently installed package versions"""
    installed = {}
    for dist in pkg_resources.working_set:
        installed[dist.project_name.lower()] = dist.version
    return installed

def compare_versions(requirements: Dict[str, str], installed: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """Compare requirements with installed versions"""
    matching = []
    mismatched = []
    missing = []
    
    for package, req_version in requirements.items():
        if package in installed:
            inst_version = installed[package]
            if req_version == "any" or req_version == inst_version:
                matching.append(f"✅ {package}: {inst_version}")
            else:
                mismatched.append(f"❌ {package}: required={req_version}, installed={inst_version}")
        else:
            missing.append(f"❌ {package}: not installed (required={req_version})")
    
    return matching, mismatched, missing

def main():
    print("🔍 Checking package version compatibility...")
    print("=" * 60)
    
    # Parse requirements.txt
    try:
        requirements = parse_requirements_file('requirements.txt')
        print(f"📋 Found {len(requirements)} packages in requirements.txt")
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return
    
    # Get installed versions
    installed = get_installed_versions()
    print(f"📦 Found {len(installed)} installed packages")
    print()
    
    # Compare versions
    matching, mismatched, missing = compare_versions(requirements, installed)
    
    # Print results
    if matching:
        print("✅ MATCHING VERSIONS:")
        for pkg in matching:
            print(f"  {pkg}")
        print()
    
    if mismatched:
        print("⚠️  VERSION MISMATCHES:")
        for pkg in mismatched:
            print(f"  {pkg}")
        print()
    
    if missing:
        print("❌ MISSING PACKAGES:")
        for pkg in missing:
            print(f"  {pkg}")
        print()
    
    # Summary
    total_required = len(requirements)
    total_matching = len(matching)
    total_mismatched = len(mismatched)
    total_missing = len(missing)
    
    print("📊 SUMMARY:")
    print(f"  Total required packages: {total_required}")
    print(f"  Matching versions: {total_matching}")
    print(f"  Version mismatches: {total_mismatched}")
    print(f"  Missing packages: {total_missing}")
    
    if total_mismatched == 0 and total_missing == 0:
        print("\n🎉 All package versions match requirements!")
    else:
        print(f"\n⚠️  {total_mismatched + total_missing} issues found")
        
        if total_mismatched > 0:
            print("\n💡 To fix version mismatches, run:")
            print("   pip install -r requirements.txt --force-reinstall")
        
        if total_missing > 0:
            print("\n💡 To install missing packages, run:")
            print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 
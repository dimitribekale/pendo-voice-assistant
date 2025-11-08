
"""
Test script to verify relative paths are working correctly.
"""

from pathlib import Path

print("="*60)
print("🧪 TESTING RELATIVE PATHS")
print("="*60)

try:
    # Test 1: Import the module
    print("\n📊 Testing pendo_core_widget.py imports...")
    from src.ui.pendo_core_widget import icon_path
    print(f"   ✅ Module imported successfully")

    # Test 2: Check if path was calculated
    print(f"\n📊 Calculated icon path:")
    print(f"   {icon_path}")

    # Test 3: Check if path is absolute (not relative)
    if icon_path.is_absolute():
        print(f"   ✅ Path is absolute (good for runtime)")
    else:
        print(f"   ⚠️  Path is relative")

    # Test 4: Check if file exists
    print(f"\n📊 Checking if icon file exists...")
    if icon_path.exists():
        print(f"   ✅ Icon file found!")
        file_size = icon_path.stat().st_size
        print(f"   📁 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    else:
        print(f"   ❌ Icon file NOT found at: {icon_path}")
        print(f"   Check if the file exists in the assets/images/ directory")
        exit(1)

    # Test 5: Check file extension
    print(f"\n📊 Validating file format...")
    if icon_path.suffix.lower() == '.png':
        print(f"   ✅ File is PNG format")
    else:
        print(f"   ⚠️  Unexpected file extension: {icon_path.suffix}")

    print("\n" + "="*60)
    print("✅ ALL PATH TESTS PASSED!")
    print("="*60)
    print("\n🎉 Your relative paths are working correctly!")
    print("The application is now portable!")

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("Make sure you're running this from the project root")
    exit(1)

except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
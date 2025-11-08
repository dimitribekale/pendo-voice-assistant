"""
  Test script to verify environment variables are loaded correctly.
"""

print("="*60)
print("🧪 TESTING CONFIGURATION")
print("="*60)

try:
    # Import config (this will load .env and validate)
    from src.config import Config

    print("\n✅ Configuration module imported successfully!")

    # Test: Verify all keys are loaded
    print("\n📊 Checking all API keys are present...")

    assert Config.OPENWEATHERMAP_API_KEY, "OPENWEATHERMAP_API_KEY is missing!"
    assert Config.NEWS_API_KEY, "NEWS_API_KEY is missing!"
    assert Config.HF_TOKEN_MODEL, "HF_TOKEN_MODEL is missing!"

    print("   ✅ All API keys present")

    # Test: Check keys are not placeholders
    print("\n📊 Checking keys are not placeholders...")

    if 'your_' in Config.OPENWEATHERMAP_API_KEY.lower():
        print("   ⚠️  WARNING: OPENWEATHERMAP_API_KEY looks like a placeholder!")
    else:
        print("   ✅ OPENWEATHERMAP_API_KEY appears valid")

    if 'your_' in Config.NEWS_API_KEY.lower():
        print("   ⚠️  WARNING: NEWS_API_KEY looks like a placeholder!")
    else:
        print("   ✅ NEWS_API_KEY appears valid")

    if 'your_' in Config.HF_TOKEN_MODEL.lower():
        print("   ⚠️  WARNING: HF_TOKEN_MODEL looks like a placeholder!")
    else:
        print("   ✅ HF_TOKEN_MODEL appears valid")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n🎉 Your configuration is working correctly!")
    print("You have successfully secured your API keys!")

except ValueError as e:
    print("\n" + "="*60)
    print("❌ CONFIGURATION ERROR")
    print("="*60)
    print(f"\n{e}\n")
    exit(1)

except Exception as e:
    print("\n" + "="*60)
    print("❌ UNEXPECTED ERROR")
    print("="*60)
    print(f"\n{e}\n")
    import traceback
    traceback.print_exc()
    exit(1)
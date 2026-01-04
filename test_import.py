"""Test if app can be imported"""
try:
    from app.main import app
    print("SUCCESS: App imported successfully")
    print(f"App type: {type(app)}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()


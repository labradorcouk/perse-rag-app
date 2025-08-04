# Utils Tests

This directory contains all test files for the utils modules.

## Test Categories

### DataFrame Tests
- `test_simple_fix.py` - Tests the simple DataFrame correction fix
- `test_fixed_dataframe_correction.py` - Tests the improved DataFrame correction logic
- `test_dataframe_fixes.py` - Tests DataFrame fixes
- `test_dataframe_fixes_final.py` - Final DataFrame fixes tests
- `test_dynamic_dataframe_fixes.py` - Tests dynamic DataFrame fixes
- `test_intelligent_fixing.py` - Tests intelligent DataFrame fixing

### Authentication Tests
- `test_auth_fixes.py` - Tests authentication fixes
- `test_auth_url.py` - Tests authentication URL generation
- `test_azure_auth.py` - Tests Azure authentication
- `test_oauth_flow.py` - Tests OAuth flow
- `test_graph_permissions.py` - Tests Graph API permissions

### Qdrant Tests
- `test_qdrant_app_issue.py` - Tests Qdrant app issues
- `test_qdrant_app_simulation.py` - Tests Qdrant app simulation
- `test_qdrant_connection.py` - Tests Qdrant connection
- `test_qdrant_debug.py` - Tests Qdrant debugging
- `test_qdrant_error_capture.py` - Tests Qdrant error capture

### Other Tests
- `test_diagnostics.py` - Tests diagnostics functionality
- `test_performance_integration.py` - Tests performance integration
- `test_pyodbc_conn.py` - Tests pyodbc connection
- `test_date_filter_handling.py` - Tests date filter handling

## Running Tests

### Run All Tests
```bash
python utils/tests/run_tests.py
```

### Run Individual Tests
```bash
python utils/tests/test_simple_fix.py
python utils/tests/test_fixed_dataframe_correction.py
# etc.
```

### Run Tests in Docker
```bash
docker-compose -f docker-compose.optimized.windows.yml exec rag-fabric-app python utils/tests/run_tests.py
```

## Test Structure

All test files follow this pattern:
1. Import required modules
2. Add project root to Python path
3. Import utils modules
4. Define test functions
5. Run tests when executed directly

## Adding New Tests

When adding new tests:
1. Place the test file in this directory
2. Use the naming convention `test_*.py`
3. Add the correct import path: `sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))`
4. Import utils modules using `from utils.module_name import ClassName`
5. Add the test to the `__init__.py` file if needed

## Notes

- Tests are designed to run both locally and in Docker
- Some tests may fail locally due to missing dependencies (expected)
- Tests should be run in the Docker environment for full functionality
- All tests import from the utils modules and test their functionality 
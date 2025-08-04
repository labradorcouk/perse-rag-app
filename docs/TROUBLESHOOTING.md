# TROUBLESHOOTING.md

## Known Issues and Solutions

### 1. ODBC/pyodbc Driver Errors
- **Symptom:** ImportError, driver not found, or authentication errors when connecting to SQL endpoint.
- **Solution:**
  - Ensure the correct ODBC driver (17 or 18) is installed and matches your Python bitness (32/64-bit).
  - Check the driver name in your connection string.
  - Verify environment variables (PATH, ODBCINI) are set correctly.
  - If the app works but the utility does not, use the CSV export workflow for fine-tuning.

### 2. Fine-Tuning Utility Issues
- **Symptom:** Errors when running `fine_tune_embedding_model.py` (e.g., ImportError for accelerate, datasets, or torch).
- **Solution:**
  - Ensure you have run `pip install -r requirements.txt`.
  - Fine-tuning utility now works only with local CSV files exported from the app's SQL Editor.
  - Do not attempt direct SQL connections in the utility script.

### 3. Environment Variable Issues
- **Symptom:** Authentication or API errors due to missing/incorrect environment variables.
- **Solution:**
  - Double-check your `.env` file for all required keys (see README and .env.example).
  - Restart your app after editing environment variables.

### 4. Dependency Issues
- **Symptom:** ImportError or missing package errors.
- **Solution:**
  - Run `pip install -r requirements.txt` to ensure all dependencies are installed.
  - For fine-tuning, ensure `accelerate` and `datasets` are present.

### 5. Workflow/Process Issues
- **Symptom:** Data not updating, export not working, or status bar not progressing.
- **Solution:**
  - Ensure you are following the recommended workflow: export Q&A/code pairs as CSV, then run the fine-tuning utility.
  - Check logs for error messages and refer to the relevant documentation section.

### 6. NameError for Utility Functions (e.g., validate_code_uses_actual_names, is_code_safe)
- **Symptom:** `NameError: name 'function_name' is not defined`
- **Solution:**
  - All utility functions are now in the `RAGUtils` class as static methods.
  - Update calls to use `RAGUtils.function_name()` instead of direct function calls.
  - Example: `validate_code_uses_actual_names()` → `RAGUtils.validate_code_uses_actual_names()`

### 7. Modular Structure Issues
- **Symptom:** Import errors for config files or utility modules.
- **Solution:**
  - Ensure all imports use the correct modular structure:
    ```python
    from config.queries import QUERIES
    from config.table_schemas import TABLE_SCHEMAS
    from config.safe_builtins import SAFE_BUILTINS
    from utils.rag_utils import RAGUtils
    ```
  - Check that all required files exist in the correct directories.

### 8. Authentication Issues
- **Symptom:** Azure authentication failures or token errors.
- **Solution:**
  - Ensure you have proper Azure AD permissions for Fabric access.
  - Check that your Azure credentials are correctly configured.
  - Verify the SQL endpoint URL and database name are correct.

### 9. Code Execution Errors
- **Symptom:** Errors when executing generated Python code.
- **Solution:**
  - The app now includes robust data type handling and error recovery.
  - Check the error messages for specific guidance on data type conversion.
  - Use the retry mechanism which automatically fixes common data type issues.

### 10. Performance Issues
- **Symptom:** Slow response times or memory issues.
- **Solution:**
  - Enable sampling for large datasets (automatic for datasets > 500 rows).
  - Reduce batch size in the UI for data fetching.
  - Use auto-fetch to only load relevant tables.
  - Check memory usage and consider increasing system RAM if needed.

### 11. Model Selection Issues
- **Symptom:** Fine-tuned models not appearing in the sidebar.
- **Solution:**
  - Ensure fine-tuned models are saved in the `models/` directory.
  - Check that the model folder contains the required files (config.json, pytorch_model.bin, etc.).
  - Restart the app after adding new models.

### 12. SQL Editor Issues
- **Symptom:** SQL queries failing or returning errors.
- **Solution:**
  - Verify SQL syntax and table/column names.
  - Check that you have proper permissions for the tables you're querying.
  - Ensure the SQL endpoint is accessible and responding.

## Debug Mode

Enable debug mode in your `.env` file:
```bash
APP_DEBUG=True
```

This will provide more detailed error messages and logging information.

## Log Files

### Audit Log
- **File:** `rag_audit_log.csv`
- **Purpose:** Tracks all RAG activities including questions, reasoning, code, and results
- **Location:** Root directory of the application

### Error Logs
- **Source:** Console output and Streamlit error messages
- **Purpose:** Detailed error information for debugging
- **Action:** Check console output for detailed error messages

## Common Error Messages and Solutions

### "OpenAI API error: name 'validate_code_uses_actual_names' is not defined"
- **Cause:** Function moved to RAGUtils class
- **Solution:** Update code to use `RAGUtils.validate_code_uses_actual_names()`

### "ModuleNotFoundError: No module named 'config'"
- **Cause:** Missing __init__.py files or incorrect import paths
- **Solution:** Ensure all directories have __init__.py files and imports use correct paths

### "Authentication failed: Invalid credentials"
- **Cause:** Azure credentials not properly configured
- **Solution:** Check Azure AD setup and permissions for Fabric access

### "Driver not found: ODBC Driver 18 for SQL Server"
- **Cause:** ODBC driver not installed or wrong version
- **Solution:** Install correct ODBC driver version matching your Python bitness

## Performance Optimization Tips

### For Large Datasets
1. **Enable Sampling:** Use the sampling feature for datasets > 500 rows
2. **Reduce Batch Size:** Lower the batch size in the UI for data fetching
3. **Use Auto-Fetch:** Only load relevant tables based on your question
4. **Optimize Queries:** Use specific date ranges and filters in your questions

### For Better Response Times
1. **Cache Results:** The app automatically caches embeddings and data
2. **Use Specific Questions:** Reference specific columns and tables in your questions
3. **Enable Code Display:** Show generated code to understand what's happening
4. **Use Visualization Keywords:** Include words like "plot", "chart", "visualize" for better results

## Getting Help

### Before Creating an Issue
1. Check this troubleshooting guide
2. Enable debug mode and check console output
3. Review the audit log for recent activities
4. Test with a simple question to isolate the issue

### When Creating an Issue
Include the following information:
- Error message and stack trace
- Steps to reproduce the issue
- Environment details (Python version, OS, etc.)
- Relevant log entries from audit log
- Screenshots if applicable

### Support Resources
- **Documentation:** Check the [docs](docs/) folder for detailed guides
- **Development Guide:** See [DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) for technical details
- **Architecture:** Review [ARCHITECTURE_DIAGRAM.md](docs/ARCHITECTURE_DIAGRAM.md) for system design

---

**Last Updated**: July 2025  
**Version**: 2.1.0 
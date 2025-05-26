# Configuration Migration Guide

## Breaking Changes in Configuration System

### Environment Variables
All environment variables now use the `FACECV_` prefix. Update your `.env` file:

**Old format:**
```
MYSQL_HOST=your-host
MYSQL_PASSWORD=your-password
```

**New format:**
```
FACECV_MYSQL_HOST=your-host
FACECV_MYSQL_PASSWORD=your-password
```

### Required MySQL Credentials
When using MySQL (`FACECV_DB_TYPE=mysql`), the following environment variables are now required:
- `FACECV_MYSQL_HOST`
- `FACECV_MYSQL_USER` 
- `FACECV_MYSQL_PASSWORD`

The application will fail to start with a clear error message if these are missing.

### Default Values
- Default MySQL host is now `localhost` instead of an empty string
- ChromaDB configuration constants have been removed and are now loaded from environment variables

### Migration Steps

1. Update your `.env` file to use the new `FACECV_` prefix for all variables
2. Ensure all required MySQL credentials are set if using MySQL
3. Update any scripts or deployment configurations that set environment variables
4. Test your application with the new configuration before deploying to production

## Configuration Reference

For a complete list of all supported environment variables, see the `.env.example` file.

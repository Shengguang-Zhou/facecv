- when you work with todo, read @docs/TODO.md, and check the checkbox once you have done. this is a migrating framework from /home/a/PycharmProjects/EurekCV. we have migrating plan at /home/a/PycharmProjects/EurekCV/plan/PLAN.md and todo.md at /home/a/PycharmProjects/EurekCV/plan/TODO.md as well.
- Migrate and support face CRUD operations for InsightFace and DeepFace APIs
  - Implement API endpoints for:
    * Video frame face sampling
    * Face database management (get, add, update, delete)
    * Face recognition by name
    * Offline face recognition
    * Webcam/RTSP stream recognition
    * Face verification
    * Face attribute analysis
  - Check InsightFace official repository and https://github.com/SthPhoenix/InsightFace-REST/tree/master/src for reference
  - Ensure compatibility with old framework APIs
- Database Configuration:
  * Database: facecv
  * Type: MySQL
  * Host: eurekailab.mysql.rds.aliyuncs.com
  * Port: 3306
  * Username: root
  * Password: Zsg20010115_
  * Configuration: Use .env file for database connection details
- when you are doing testing, if its a temporary test, remove it after test, if its reusable or later will be used, put under test/ folder, same applied for log and sqlite.
- when you are testing face, you can use this folder for your test: /home/a/PycharmProjects/EurekCV/dataset/faces
- when testing missing dependency, use .venv to install
- please follow agile development style, always follow dev -> test -> fix  -> dev loop when doing work. When you are testing fastapi, for tmp test,  use curl, for reusable test use .http file. Make your files manageable in test folder if its a reusable file. For temporary file, remember clean i  up after testing
- always start server in reload mode if needed
- when you are unsure about a repository, use context7 mcp to help you
- when you are writing a project, use nx mcp to help you understand structure
// routes/uploadRoutes.js
const express = require("express");
const router = express.Router();
const uploadController = require("../controllers/uploadController"); // Path to your upload controller
// Assuming you have multer configured in server.js and pass it here, or configure it here
const multer = require("multer"); // Import multer
// Define storage for multer in this file or import from a config file
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/"); // Make sure this directory exists
  },
  filename: function (req, file, cb) {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});
const upload = multer({ storage: storage });

// File upload route
// Apply multer middleware directly to this route
router.post("/", upload.array("files"), uploadController.uploadFiles); // Correct for /api/upload

module.exports = router;

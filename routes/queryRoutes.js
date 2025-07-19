// routes/queryRoutes.js
const express = require("express");
const router = express.Router();
const queryController = require("../controllers/queryController"); // Path to your query controller
const { protect } = require("./../middleware/authMiddleware");
// If this route also accepts file uploads, include multer here
const multer = require("multer");
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});
const upload = multer({ storage: storage });

// AI query route
// Apply multer middleware if files are expected here too
router.post("/", upload.array("files"), protect, queryController.sendPrompt); // Correct for /api/query

module.exports = router;

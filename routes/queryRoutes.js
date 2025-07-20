// routes/queryRoutes.js
const express = require("express");
const router = express.Router();
const queryController = require("../controllers/queryController"); // Path to your query controller
const { protect } = require("./../middleware/authMiddleware");
// If this route also accepts file uploads, include multer here

// AI query route
// Apply multer middleware if files are expected here too
router.post("/", protect, queryController.sendPrompt); // Correct for /api/query

module.exports = router;

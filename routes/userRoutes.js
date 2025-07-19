// routes/userRoutes.js
const express = require("express");
const router = express.Router();
const userController = require("../controllers/userController"); // Path to your user controller
// You might also need protect middleware if certain user routes are private
// const { protect } = require("../controllers/authController");

// User authentication/management routes
router.post("/signup", userController.signUp);
router.post("/signIn", userController.signUser);
// Correct for /api/user/signIn
// Add other user routes here, e.g.,
// router.get("/profile", protect, userController.getUserProfile);

module.exports = router;

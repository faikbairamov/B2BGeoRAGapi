// index.js (or server.js, depending on your main file name)
const express = require("express");
const cors = require("cors");
const morgan = require("morgan"); // optional for logging
require("dotenv").config(); // Load environment variables from .env

// Database connection
const connectDB = require("./db"); // Assuming your db.js is in a 'config' folder
connectDB(); // Call the database connection function

// Import individual route files
const userRoutes = require("./routes/userRoutes");
const uploadRoutes = require("./routes/uploadRoutes"); // NEW import
const queryRoutes = require("./routes/queryRoutes"); // NEW import

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json()); // Parse JSON bodies
app.use(express.urlencoded({ extended: true })); // For form-data (e.g., from simple HTML forms)
app.use(morgan("dev")); // Log requests (e.g., GET / 200 5ms)

// Apply routes
// Each router is now mounted to its specific base path
app.use("/api/user", userRoutes); // Handles /api/user/signIn etc.
app.use("/api/upload", uploadRoutes); // Handles /api/upload
app.use("/api/query", queryRoutes); // Handles /api/query

// Health check
app.get("/", (req, res) => {
  res.send("✅ RAG Backend is running!");
});

// Global Error Handler (Highly Recommended)
// This should be the last app.use() in your middleware stack
app.use((err, req, res, next) => {
  console.error(err.stack); // Log the error stack for debugging
  // Determine status code (default to 500 Internal Server Error)
  const statusCode = err.statusCode || 500;
  // Send a standardized error response
  res.status(statusCode).json({
    success: false,
    message: err.message || "Internal Server Error",
    // In development, you might send the full error for debugging
    // error: process.env.NODE_ENV === 'development' ? err : {}
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server listening at http://localhost:${PORT}`);
});

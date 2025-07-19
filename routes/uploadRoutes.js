// routes/uploadRoutes.js
const express = require("express");
const router = express.Router();
const uploadController = require("../controllers/uploadController"); // Path to your upload controller
// Assuming you have multer configured in server.js and pass it here, or configure it here
const multer = require("multer"); // Import multer
// Define storage for multer in this file or import from a config file


const upload = multer({ dest: 'uploads/' });

router.post(
  '/',
  upload.array('files'),
  uploadController.uploadFiles
);


module.exports = router;

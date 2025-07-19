const User = require("../models/users"); // Make sure this path is correct
const asyncHandler = require("express-async-handler"); // If not installed, npm i express-async-handler

// Helper to send JWT token in response
const sendTokenResponse = (user, statusCode, res) => {
  const token = user.getSignedJwtToken(); // This method should be defined in your userModel

  const options = {
    expires: new Date(
      Date.now() + process.env.JWT_COOKIE_EXPIRE * 24 * 60 * 60 * 1000
    ),
    httpOnly: true,
  };

  if (process.env.NODE_ENV === "production") {
    options.secure = true;
  }

  res
    .status(statusCode)
    .cookie("token", token, options)
    .json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        username: user.username,
        email: user.email,
      },
    });
};

/**
 * @desc    Sign in user & get token
 * @route   POST /api/signIn
 * @access  Public
 */
exports.signUp = asyncHandler(async (req, res, next) => {
  const { name, username, email, password, company } = req.body; // Extract user data from request body

  // Basic validation (more comprehensive validation should be done with a library like Joi or Express-validator)
  if (!name || !username || !email || !password || !company) {
    return res
      .status(400)
      .json({ success: false, error: "Please fill in all required fields." });
  }

  try {
    // 1. Create a new user in the database
    // The pre-save hook in your userModel will automatically hash the password
    const user = await User.create({
      name,
      username,
      email,
      password,
      company,
    });

    // 2. Send JWT token upon successful registration
    sendTokenResponse(user, 201, res); // 201 status code for successful creation
  } catch (error) {
    // Handle specific errors like duplicate email/username
    if (error.code === 11000) {
      // MongoDB duplicate key error code
      // Check if the duplicate is email or username
      const field = Object.keys(error.keyValue)[0];
      const message = `A user with that ${field} already exists.`;
      return res.status(400).json({ success: false, error: message });
    }
    // For other validation errors (e.g., minlength), Mongoose handles them
    // The global error handler should catch these.
    next(error); // Pass other errors to the global error handling middleware
  }
});
exports.signUser = asyncHandler(async (req, res, next) => {
  const { email, password } = req.body;

  // 1. Validate email and password
  if (!email || !password) {
    // You should use a custom error handler here for better error messages
    return res
      .status(400)
      .json({ success: false, error: "Please provide an email and password." });
  }

  // 2. Check for user (explicitly select password)
  const user = await User.findOne({ email });

  if (!user) {
    return res
      .status(401)
      .json({ success: false, error: "Invalid credentials." });
  }

  // 3. Check if password matches
  const isMatch = await user.comparePassword(password); // This method should be defined in your userModel

  if (!isMatch) {
    return res
      .status(401)
      .json({ success: false, error: "Invalid credentials." });
  }

  // 4. Send JWT token
  sendTokenResponse(user, 200, res);
});

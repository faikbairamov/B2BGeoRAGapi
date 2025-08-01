const jwt = require("jsonwebtoken");
const asyncHandler = require("express-async-handler");
const User = require("../models/users");
exports.protect = asyncHandler(async (req, res, next) => {
  let token;

  if (
    req.headers.authorization &&
    req.headers.authorization.startsWith("Bearer")
  ) {
    console.log(req.headers.authorization); // Log the authorization header for debugging
    // Extract token from 'Bearer <token>' format
    try {
      token = req.headers.authorization.split(" ")[1];

      // Verify token
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      console.log(decoded)

      // Attach user to the request object (without password)
      // The decoded object typically contains { id: user._id, iat: ..., exp: ... }
      const user = await User.findById(decoded.id)

      console.log(user)
      req.user = user

      if (!req.user) {
        return res
          .status(401)
          .json({ success: false, message: "User not found for this token." });
      }

      next(); // Proceed to the next middleware or route handler
    } catch (error) {
      console.error(error);
      return res
        .status(401)
        .json({ success: false, message: "Not authorized, token failed." });
    }
  }

  if (!token) {
    return res
      .status(401)
      .json({ success: false, message: "Not authorized, no token." });
  }
});

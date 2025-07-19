//email, saxeli, gvari. company
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const jsonwebtoken = require("jsonwebtoken");

// User Schema
const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true,
  },
  username: {
    type: String,
    required: true,
    trim: true,
  },

  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true,
  },
  password: {
    type: String,
    required: true,
    minlength: 6,
  },
});

userSchema.pre("save", async function (next) {
  if (!this.isModified("password")) {
    // Only hash if the password field is new or has been modified
    return next();
  }
  const salt = await bcrypt.genSalt(10); // Generate a salt with 10 rounds
  this.password = await bcrypt.hash(this.password, salt); // Hash the password
  next();
});

// 🔑 Sign JWT and return token for the user 🚀
userSchema.methods.getSignedJwtToken = function () {
  return jsonwebtoken.sign(
    { id: this._id }, // Payload: user's ID
    process.env.JWT_SECRET, // Your secret key from .env
    {
      expiresIn: process.env.JWT_EXPIRE, // Token expiration from .env
    }
  );
};

// ✅ Match user entered password to hashed password in database 🤝
userSchema.methods.comparePassword = async function (enteredPassword) {
  // `this.password` refers to the hashed password stored in the database
  // `enteredPassword` is the plain text password provided by the user
  return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model("User", userSchema);

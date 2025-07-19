// db.js (CommonJS compatible)
const mongoose = require("mongoose"); // Use require() for mongoose

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGO_URI);
    console.log(`MongoDB Connected: ${conn.connection.host} 🥳`); // Added host for better logging
  } catch (error) {
    console.error(`Error connecting to MongoDB: ${error.message} 😵‍💫`);
    process.exit(1); // Exit process with failure
  }
};

module.exports = connectDB; // Correctly exports the function using CommonJS

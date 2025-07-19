const pdfParse = require('pdf-parse');
const fs = require('fs');

exports.uploadFiles = async (req, res) => {
  try {
    const results = [];

    for (const file of req.files) {
      const dataBuffer = fs.readFileSync(file.path);
      const data = await pdfParse(dataBuffer);

      results.push({
        filename: file.originalname,
        text: data.text,
      });

      fs.unlinkSync(file.path); // delete temp file
    }

    res.json({ results });
  } catch (err) {
    console.error('Error extracting PDF text:', err);
    res.status(500).json({ error: 'Failed to process PDF files' });
  }
};
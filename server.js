const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');
const path = require('path');
const cors = require('cors');
const csv = require('csv-parser');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.post('/upload', upload.single('file'), (req, res) => {
    console.log("file upload request received");
    const filePath = req.file.path;
    const results = [];

    fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => {
            const columns = Object.keys(results[0]);
            // Save the data to 'temp.csv' for the Python script to access
            const tempFilePath = path.join(__dirname, 'uploads/temp.csv');
            fs.writeFileSync(tempFilePath, fs.readFileSync(filePath));
            res.json({ columns });
        });
});

app.post('/train', (req, res) => {
    console.log("train request received");
    const { features, target, modelType } = req.body;
    const options = {
        args: [features.join(','), target, modelType],
        pythonOptions: ['-u'], // get print results in real-time
        scriptPath: path.join(__dirname) // Ensure Python script is found
    };

    let pyshell = new PythonShell('train_model.py', options);

    let output = "";

    pyshell.on('message', function (message) {
        console.log("Python script output:", message);
        if(message[0] === '{' && message[message.length - 1] === '}'){
            output += message;
        }
        
    });

    pyshell.on('stderr', function (stderr) {
        console.error("Python script error:", stderr);
    });

    pyshell.end(function (err, code, signal) {
        if (err) {
            console.error("Error during training:", err);
            return res.status(500).json({ error: err.message });
        }
        console.log("Training completed with exit code:", code);

        try {
            console.log("Parsing Python script output");
            console.log(output);
            const result = JSON.parse(output);
            console.log(result);
            res.json(result);
        } catch (parseError) {
            console.error("Error parsing Python script output:", parseError);
            res.status(500).json({ error: "Error parsing Python script output" });
        }
    });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});

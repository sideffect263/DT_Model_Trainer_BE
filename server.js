const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');
const path = require('path');
const cors = require('cors');
const csv = require('csv-parser');
const fs = require('fs');

const { v4: uuidv4 } = require('uuid'); // Import uuid for unique session IDs

const app = express();
app.use(cors());
app.use(bodyParser.json());

const upload = multer({ dest: 'uploads/' });

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.post('/upload', upload.single('file'), (req, res) => {
    const sessionId = uuidv4(); // Generate a unique session ID
    console.log(`file upload request received for session: ${sessionId}`);
    const filePath = req.file.path;
    const results = [];

    fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => {
            const columns = Object.keys(results[0]);
            const sessionDir = path.join(__dirname, 'uploads', sessionId);
            if (!fs.existsSync(sessionDir)) {
                fs.mkdirSync(sessionDir);
            }
            const tempFilePath = path.join(sessionDir, 'temp.csv');
            fs.writeFileSync(tempFilePath, fs.readFileSync(filePath));
            res.json({ sessionId, columns });
        });
});

app.post('/train', (req, res) => {
    const { sessionId, features, target, modelType } = req.body;
    console.log(`train request received for session: ${sessionId}`);
    const options = {
        args: [features.join(','), target, modelType, sessionId],
        pythonOptions: ['-u'],
        scriptPath: path.join(__dirname)
    };

    let pyshell = new PythonShell('train_model.py', options);

    let output = "";

    pyshell.on('message', function (message) {
        console.log("Python script output:", message);
        if (message[0] === '{' && message[message.length - 1] === '}') {
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

app.post('/predict', (req, res) => {
    const { sessionId, features, data } = req.body;
    console.log(`predict request received for session: ${sessionId}`);
    const options = {
        args: [features.join(','), JSON.stringify(data), sessionId],
        pythonOptions: ['-u'],
        scriptPath: path.join(__dirname)
    };

    let pyshell = new PythonShell('predict.py', options);

    let output = "";

    pyshell.on('message', function (message) {
        console.log("Python script output:", message);
        if (message[0] === '{' && message[message.length - 1] === '}') {
            output += message;
        }
    });

    pyshell.on('stderr', function (stderr) {
        console.error("Python script error:", stderr);
    });

    pyshell.end(function (err, code, signal) {
        if (err) {
            console.error("Error during prediction:", err);
            return res.status(500).json({ error: err.message });
        }
        console.log("Prediction completed with exit code:", code);

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

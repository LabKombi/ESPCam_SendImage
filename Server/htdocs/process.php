<?php
header('Content-Type: application/json');

// Aktifkan error reporting untuk debugging
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Logging untuk debugging
error_log("Request method: " . $_SERVER['REQUEST_METHOD']);
error_log("Files received: " . json_encode($_FILES));
error_log("Post data: " . json_encode($_POST));

// Periksa metode request
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(['error' => 'Invalid request method.']);
    exit;
}

// Periksa apakah file diterima
if (!isset($_FILES['image'])) {
    echo json_encode(['error' => 'No image file provided.']);
    exit;
}

$file = $_FILES['image'];
$tempPath = $file['tmp_name'];
$fileName = $file['name'];

// Periksa apakah file diunggah dengan benar
if (!is_uploaded_file($tempPath)) {
    echo json_encode(['error' => 'File upload error.']);
    exit;
}

// Fungsi untuk memanggil Python script
function runPythonPredict($imagePath) {
    $pythonScriptPath = "../app/predictor.py";
    $command = escapeshellcmd("python $pythonScriptPath --image $imagePath");
    $output = [];
    $returnCode = 0;

    exec($command, $output, $returnCode);

    if ($returnCode !== 0) {
        return ["error" => "Error executing Python script."];
    }

    // Gabungkan output jika lebih dari satu baris
    $resultJson = implode("\n", $output);

    // Decode JSON dari output Python
    return json_decode($resultJson, true);
}

// Jalankan prediksi menggunakan Python
$resultData = runPythonPredict($tempPath);

// Hapus file sementara
if (file_exists($tempPath)) {
    unlink($tempPath);
}

// Periksa hasil dari Python
if (isset($resultData['error'])) {
    echo json_encode(['error' => $resultData['error']]);
    exit;
}

// Kirim hasil prediksi sebagai respons JSON
echo json_encode($resultData);

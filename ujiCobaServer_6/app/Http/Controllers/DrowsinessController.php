<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Log;

class DrowsinessController extends Controller
{
    public function predict(Request $request)
    {
        ini_set('max_execution_time', 300); // 300 detik = 5 menit

        // Validasi input
        $request->validate([
            'image' => 'required|image',
        ]);

        // Simpan gambar ke folder sementara
        $imagePath = $request->file('image')->store('temp');
        $absoluteImagePath = Storage::path($imagePath);

        // Jalankan skrip Python untuk prediksi
        $pythonScriptPath = storage_path('app/python/app.py');
        $command = escapeshellcmd("python \"$pythonScriptPath\" --image \"$absoluteImagePath\"");

        $output = shell_exec($command);

        // Log output untuk debugging
        Log::info('Python Output: ' . $output);

        // Hapus file sementara
        Storage::delete($imagePath);

        // Periksa apakah output kosong
        if (!$output) {
            return response()->json(['error' => 'No output from Python script'], 500);
        }

        // Menghapus karakter kontrol ANSI dan newline ekstra
        $output = preg_replace('/\x1b\[[0-9;]*m/', '', $output); // Menghapus semua karakter ANSI
        $output = preg_replace('/\s+/', ' ', $output);  // Menghapus karakter spasi berlebih dan newline
        $output = trim($output); // Menghapus spasi di awal dan akhir

        // Regex untuk menangkap prediction dan confidence dari output
        if (preg_match('/"prediction":\s*"([^"]+)".*"confidence":\s*([\d\.]+)/', $output, $matches)) {
            $prediction = $matches[1];  // Ambil prediction
            $confidence = $matches[2];  // Ambil confidence
        } else {
            return response()->json(['error' => 'Failed to extract prediction and confidence'], 500);
        }

        // Kembalikan hasil prediksi dalam format JSON
        return response()->json([
            'prediction' => $prediction,
            'confidence' => (float) $confidence
        ]);
    }
}
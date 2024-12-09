<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

use App\Http\Controllers\DrowsinessController;

Route::post('/predict', [DrowsinessController::class, 'predict']);



<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('images', function (Blueprint $table) {
            $table->id();
            $table->string('title')->nullable();
            $table->string('image_path'); // مسیر فایل تصویر
            $table->string('image_url')->nullable(); // URL تصویر
            $table->integer('question_count')->default(3); // تعداد سوالات برای این تصویر
            $table->enum('type', ['workbench', 'level_test'])->default('workbench'); // نوع: میز کار یا آزمون تعیین سطح
            $table->boolean('is_active')->default(true);
            $table->foreignId('created_by')->nullable()->constrained('users')->onDelete('set null');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('images');
    }
};

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
        Schema::create('questions', function (Blueprint $table) {
            $table->id();
            $table->foreignId('image_id')->constrained()->onDelete('cascade');
            $table->text('question_text'); // متن سوال
            $table->json('options')->nullable(); // گزینه‌ها به صورت JSON برای آزمون تعیین سطح
            $table->integer('correct_option')->nullable(); // ایندکس گزینه صحیح (برای آزمون تعیین سطح)
            $table->enum('type', ['open_ended', 'multiple_choice'])->default('open_ended');
            $table->integer('accuracy_score')->default(5); // امتیاز دقت (حداکثر 5)
            $table->boolean('is_control_question')->default(false); // سوال کنترلی برای بررسی کیفیت
            $table->integer('order')->default(0); // ترتیب سوال
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('questions');
    }
};

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
        Schema::create('quiz_attempts', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->integer('attempt_number')->default(1); // شماره تلاش
            $table->json('answers')->nullable(); // پاسخ‌ها به صورت JSON
            $table->integer('total_score')->default(0); // امتیاز کل
            $table->integer('accuracy_score')->default(0); // امتیاز دقت
            $table->integer('speed_score')->default(0); // امتیاز سرعت
            $table->integer('question_count')->default(12); // تعداد سوالات
            $table->string('calculated_level')->nullable(); // سطح محاسبه شده
            $table->boolean('is_completed')->default(false);
            $table->timestamp('started_at')->nullable();
            $table->timestamp('completed_at')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('quiz_attempts');
    }
};

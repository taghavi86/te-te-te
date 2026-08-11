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
        Schema::create('user_levels', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->foreignId('level_id')->constrained()->onDelete('cascade');
            $table->integer('current_score')->default(0); // امتیاز فعلی کاربر
            $table->integer('total_tokens')->default(0); // مجموع توکن‌های کسب شده
            $table->integer('confirmed_tokens')->default(0); // توکن‌های تایید شده
            $table->timestamp('last_quiz_attempt_at')->nullable(); // آخرین زمان آزمون
            $table->integer('quiz_attempts_count')->default(0); // تعداد دفعات شرکت در آزمون
            $table->timestamp('next_quiz_allowed_at')->nullable(); // زمان مجاز بعدی برای آزمون
            $table->boolean('is_active')->default(true);
            $table->timestamps();
            
            $table->unique('user_id');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('user_levels');
    }
};

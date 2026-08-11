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
        Schema::create('weekly_rankings', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->integer('year'); // سال
            $table->integer('week'); // شماره هفته
            $table->integer('rank')->nullable(); // رتبه
            $table->integer('total_tokens')->default(0); // مجموع توکن‌ها
            $table->integer('completed_tasks')->default(0); // تعداد وظایف انجام شده
            $table->integer('accuracy_rate')->default(0); // نرخ دقت (درصد)
            $table->integer('speed_score')->default(0); // امتیاز سرعت
            $table->boolean('reward_paid')->default(false); // آیا پاداش پرداخت شده
            $table->integer('reward_amount')->default(0); // مبلغ پاداش
            $table->timestamps();
            
            $table->unique(['user_id', 'year', 'week']);
            $table->index(['year', 'week']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('weekly_rankings');
    }
};

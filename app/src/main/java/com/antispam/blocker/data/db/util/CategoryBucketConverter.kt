package com.antispam.blocker.data.db.util

import androidx.room.TypeConverter
import com.antispam.blocker.data.db.entity.NotificationEvent

class CategoryBucketConverter {

    @TypeConverter
    fun fromCategoryBucket(bucket: NotificationEvent.CategoryBucket): String = bucket.name

    @TypeConverter
    fun toCategoryBucket(name: String): NotificationEvent.CategoryBucket =
        NotificationEvent.CategoryBucket.valueOf(name)
}

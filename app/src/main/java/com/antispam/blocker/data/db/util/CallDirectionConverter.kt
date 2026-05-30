package com.antispam.blocker.data.db.util

import androidx.room.TypeConverter
import com.antispam.blocker.data.db.entity.CallEvent

class CallDirectionConverter {

    @TypeConverter
    fun fromDirection(direction: CallEvent.Direction): String = direction.name

    @TypeConverter
    fun toDirection(name: String): CallEvent.Direction = CallEvent.Direction.valueOf(name)
}

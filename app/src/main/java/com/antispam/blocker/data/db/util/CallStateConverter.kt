package com.antispam.blocker.data.db.util

import androidx.room.TypeConverter
import com.antispam.blocker.data.db.entity.CallEvent

class CallStateConverter {

    @TypeConverter
    fun fromCallState(state: CallEvent.CallState): String = state.name

    @TypeConverter
    fun toCallState(name: String): CallEvent.CallState = CallEvent.CallState.valueOf(name)
}

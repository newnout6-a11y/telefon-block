package com.antispam.blocker.data.db.util

import androidx.room.TypeConverter
import com.antispam.blocker.domain.detector.Verdict

class VerdictConverter {

    @TypeConverter
    fun fromVerdict(verdict: Verdict): String = verdict.name

    @TypeConverter
    fun toVerdict(name: String): Verdict = Verdict.valueOf(name)
}

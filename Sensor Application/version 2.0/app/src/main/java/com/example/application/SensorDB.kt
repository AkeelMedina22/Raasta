package com.example.application

import android.hardware.Sensor
import androidx.lifecycle.LiveData
import androidx.room.*

@Entity(tableName = "sensor_db")
data class SensorDB(
    @PrimaryKey(autoGenerate = true) val id : Int,
    @ColumnInfo(name = "AndroidID") val a_id: String?,
    @ColumnInfo(name = "SessionID") val s_id: String?,
    @ColumnInfo(name = "TimeStamp") val ts: String?,
    @ColumnInfo(name = "Label") val lab: String?,
    @ColumnInfo(name = "Latitude") val lat: String?,
    @ColumnInfo(name = "Longitude") val long: String?,
    @ColumnInfo(name = "AccelerometerX") val accx: String?,
    @ColumnInfo(name = "AccelerometerY") val accy: String?,
    @ColumnInfo(name = "AccelerometerZ") val accz: String?,
    @ColumnInfo(name = "GyroscopeX") val gyrx: String?,
    @ColumnInfo(name = "GyroscopeY") val gyry: String?,
    @ColumnInfo(name = "GyroscopeZ") val gyrz: String?,
)

@Dao
interface SensorDao{

    @Insert
    fun insert(sensor : SensorDB)

    @Delete
    fun delete(sensor: SensorDB)

    @Query("SELECT * FROM sensor_db")
    fun getAllData(): List<SensorDB>

    @Query("SELECT * FROM sensor_db WHERE SessionID LIKE :s_id")
    fun getData(s_id : String): List<SensorDB>

    @Query("SELECT COUNT(*) FROM sensor_db")
    fun countRows(): Int
}

@Database(entities = [SensorDB::class], version = 1)
abstract class SensorDatabase : RoomDatabase(){
    abstract fun s_doa():SensorDao
}
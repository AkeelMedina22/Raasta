package com.example.kaavish

// To-do:
// Labelling data, changing database structure, changing code to match it, requiring wifi connection? what happens with no wifi.

import android.Manifest
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.*
import com.google.firebase.database.DatabaseReference
import com.google.firebase.database.FirebaseDatabase
import java.security.AccessController.getContext
import java.text.SimpleDateFormat
import java.util.*
import android.provider.Settings
import android.annotation.SuppressLint
import android.location.Location
import android.location.LocationManager
import android.util.Log
import androidx.core.location.LocationManagerCompat.isLocationEnabled

class MainActivity : AppCompatActivity(), SensorEventListener {

    companion object {
        private const val UPDATE_INTERVAL_IN_MILLISECONDS = 500L
        private const val FASTEST_UPDATE_INTERVAL_IN_MILLISECONDS =
            UPDATE_INTERVAL_IN_MILLISECONDS / 2
    }
    lateinit var fusedLocationProviderClient: FusedLocationProviderClient
    lateinit var locationRequest : LocationRequest
    lateinit var locationResult : Location
    var latitude : Double = 0.0
    var longitude : Double = 0.0

    private lateinit var mSensorManager : SensorManager
    private var mAccelerometer : Sensor ?= null
    private var mGyroscope : Sensor ?= null
    private var resume = false

    // added this line - abeer
    //private val android_id = Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)
    private lateinit var database: DatabaseReference

    override fun onCreate(savedInstanceState: Bundle?) {


        fusedLocationProviderClient = LocationServices.getFusedLocationProviderClient(this)

        requestWindowFeature(Window.FEATURE_NO_TITLE)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN)

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mSensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        database = FirebaseDatabase.getInstance().getReference("sensor-data")

        getLastLocation()

    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        print("accuracy changed")
    }

    override fun onSensorChanged(event: SensorEvent?) {

        if (event != null && resume) {

            val date = Date()
            val formatting = SimpleDateFormat("yyyymmddhhmmss")
            val formatteddate = formatting.format(date)

//        val uid = android_id + "-" + formatteddate
//        println("uid")

            //add this to database
            database.child(formatteddate).child("timestamp").setValue(formatteddate).addOnSuccessListener {
                Toast.makeText(this, "Succesfully saved!", Toast.LENGTH_SHORT).show()
            }.addOnFailureListener{
                Toast.makeText(this, "Failed!", Toast.LENGTH_SHORT).show()
            }

            try{
                latitude = locationResult.latitude
                longitude = locationResult.longitude
                findViewById<TextView>(R.id.longt).text = longitude.toString()
                findViewById<TextView>(R.id.lat).text = latitude.toString()
                database.child(formatteddate).child("latitude").setValue(latitude.toString())
                database.child(formatteddate).child("longitude").setValue(longitude.toString())
            }
            catch(e: Exception){
                println("failed")
            }


            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                val accX = event.values[0]
                val temp01:Double = String.format("%.3f", accX).toDouble()
                val temp02:Double = String.format("%.2f", temp01).toDouble()
                val temp03:Double = String.format("%.1f", temp02).toDouble()
                findViewById<TextView>(R.id.acc_X).text = temp03.toString()

                val accY = event.values[1]
                val temp11:Double = String.format("%.3f", accY).toDouble()
                val temp12:Double = String.format("%.2f", temp11).toDouble()
                val temp13:Double = String.format("%.1f", temp12).toDouble()
                findViewById<TextView>(R.id.acc_Y).text = temp13.toString()

                val accZ = event.values[2]
                val temp21:Double = String.format("%.3f", accZ).toDouble()
                val temp22:Double = String.format("%.2f", temp21).toDouble()
                val temp23:Double = String.format("%.1f", temp22).toDouble()
                findViewById<TextView>(R.id.acc_Z).text = temp23.toString()

                database.child(formatteddate).child("accelerometer-x").setValue(temp03.toString())
                database.child(formatteddate).child("accelerometer-y").setValue(temp13.toString())
                database.child(formatteddate).child("accelerometer-z").setValue(temp23.toString())

            }

            if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
                val accX = event.values[0]
                val temp01:Double = String.format("%.3f", accX).toDouble()
                val temp02:Double = String.format("%.2f", temp01).toDouble()
                val temp03:Double = String.format("%.1f", temp02).toDouble()
                findViewById<TextView>(R.id.gyro_x).text = temp03.toString()

                val accY = event.values[1]
                val temp11:Double = String.format("%.3f", accY).toDouble()
                val temp12:Double = String.format("%.2f", temp11).toDouble()
                val temp13:Double = String.format("%.1f", temp12).toDouble()
                findViewById<TextView>(R.id.gyro_y).text = temp13.toString()

                val accZ = event.values[2]
                val temp21:Double = String.format("%.3f", accZ).toDouble()
                val temp22:Double = String.format("%.2f", temp21).toDouble()
                val temp23:Double = String.format("%.1f", temp22).toDouble()
                findViewById<TextView>(R.id.gyro_z).text = temp23.toString()

                database.child(formatteddate).child("gyroscope-x").setValue(temp03.toString())
                database.child(formatteddate).child("gyroscope-y").setValue(temp13.toString())
                database.child(formatteddate).child("gyroscope-z").setValue(temp23.toString())
            }
        }
    }

    override fun onResume() {
        super.onResume()
        mSensorManager.registerListener(this, mAccelerometer, 500000, 500000)
        mSensorManager.registerListener(this, mGyroscope, 500000, 500000)
    }

    override fun onPause() {
        super.onPause()
        mSensorManager.unregisterListener(this)
    }

    fun resumeReading(view: View) {
        this.resume = true
    }

    fun pauseReading(view: View) {
        this.resume = false
    }

    // LOCATION STUFF

    @SuppressLint("MissingPermission")
    private fun getLastLocation() {
        if(CheckPermission()){
            if(isLocationEnabled()){

                fusedLocationProviderClient.lastLocation.addOnCompleteListener{task ->
                    var location = task.result
                    if(location == null){

                        getNewLocation()
                    }else {
                        locationResult = location
                    }
                }

            }else{
                Toast.makeText(this, "Location service not enabled", Toast.LENGTH_SHORT).show()
            }

        }else{
            RequestPermission()
        }
    }

    private fun CheckPermission() : Boolean {
        if( (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
            || (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) ){
            return true
        }

        return false
    }

    private fun RequestPermission() {

        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ), 1000
        )

    }

    private fun isLocationEnabled() : Boolean{
        var locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        return locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) || locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == 1000){
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                println("Permission granted")
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun getNewLocation(){
        locationRequest = LocationRequest()
        locationRequest.priority = LocationRequest.PRIORITY_HIGH_ACCURACY
        locationRequest.interval = 0
        locationRequest.fastestInterval = 0
        locationRequest.numUpdates = 2
        fusedLocationProviderClient!!.requestLocationUpdates(
            locationRequest, locationCallback, null
        )

    }

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(p0: LocationResult?){
            var lastLocation = p0?.lastLocation

            if (lastLocation != null) {
                locationResult = lastLocation
            }
        }

    }
}
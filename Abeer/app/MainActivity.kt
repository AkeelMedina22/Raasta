package com.example.workk
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
import java.text.SimpleDateFormat
import java.util.*
import android.provider.Settings
import android.annotation.SuppressLint
import android.location.Location
import android.location.LocationManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.widget.Button
import com.google.firebase.database.*

class MainActivity : AppCompatActivity(), SensorEventListener {

    // location initialization
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

    // sensor initialization
    private lateinit var mSensorManager : SensorManager
    private var mAccelerometer : Sensor ?= null
    private var mGyroscope : Sensor ?= null

    // start button flag
    private var resume = false

    // category button click counts
    private var pothole_btn_click = 0
    private var speedbreaker_btn_click = 0
    private var traffic_btn_click = 0
    private var schange_btn_click = 0
    private var broad_btn_click = 0

    // category labels
    private var label = "Normal Road"
    private var pothole_l = "Pothole"
    private var speedbreaker_l = "Speedbreaker"
    private var traffic_l = "Traffic"
    private var schange_l = "Sudden Change"
    private var broad_l = "Bad Road"

    private var android_id : String = ""
    private var session_id : String = ""

    var count = 0
    private var myList = mutableListOf<MutableList<String>>()

    // database initialization
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
        mSensorManager.registerListener(this, mAccelerometer, 500000, 500000)
        mSensorManager.registerListener(this, mGyroscope, 500000, 500000)

        database = FirebaseDatabase.getInstance().getReference("sensor-data")

        android_id = Settings.Secure.getString(this.contentResolver, Settings.Secure.ANDROID_ID)

        session_id = UUID.randomUUID().toString()

        //check if session_id exists in the database
//        database.addValueEventListener(object : ValueEventListener {
//            override fun onDataChange(dataSnapshot : DataSnapshot)
//            {
//                while(dataSnapshot.child(session_id).exists())
//                {
//                    session_id = UUID.randomUUID().toString()
//                }
//            }
//            override fun onCancelled(error : DatabaseError)
//            {}
//        })

        //getLastLocation()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        print("accuracy changed")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        count += 1
        println(count)

        getLastLocation()

        var sensor_label = ""
        var accelerometer_x = "0.0"
        var accelerometer_y = "0.0"
        var accelerometer_z = "0.0"

        var gyroscope_x = "0.0"
        var gyroscope_y = "0.0"
        var gyroscope_z = "0.0"

        // when start button is pressed, data will start getting collected
        if (event != null && resume) {
            getLastLocation()

            val date = Date()
            val formatting = SimpleDateFormat("yyyyMMddHHmmss-SSS")
            val formatteddate = formatting.format(date)

            //add this to database
//            database.child(session_id).child(formatteddate).child("timestamp").setValue(formatteddate)
//            database.child(session_id).child(formatteddate).child("android-id").setValue(android_id)

            val pothole_btn = findViewById<Button>(R.id.pothole)
            val speedbreaker_btn = findViewById<Button>(R.id.speedbreaker)
            val traffic_btn = findViewById<Button>(R.id.traffic)
            val bad_road_btn = findViewById<Button>(R.id.badroad)
            val sudden_change_btn = findViewById<Button>(R.id.suddenchange)

            pothole_btn.setOnClickListener(object : View.OnClickListener {
                override fun onClick(view: View)
                {
                    pothole_btn_click = pothole_btn_click + 1
                    if (pothole_btn_click == 1)
                    {
                        sensor_label = pothole_l
//                        database.child(session_id).child(formatteddate).child("label").setValue(pothole_l)
                        speedbreaker_btn.setEnabled(false)
                        traffic_btn.setEnabled(false)
                        bad_road_btn.setEnabled(false)
                        sudden_change_btn.setEnabled(false)
                    }
                    else
                    {
                        sensor_label = label
//                        database.child(session_id).child(formatteddate).child("label").setValue(label)
                        pothole_btn_click = 0
                        speedbreaker_btn.setEnabled(true)
                        traffic_btn.setEnabled(true)
                        bad_road_btn.setEnabled(true)
                        sudden_change_btn.setEnabled(true)
                    }
                }
            })

            speedbreaker_btn.setOnClickListener(object : View.OnClickListener
            {
                override fun onClick(view : View)
                {
                    speedbreaker_btn_click = speedbreaker_btn_click + 1
                    if (speedbreaker_btn_click == 1)
                    {
                        sensor_label = speedbreaker_l
//                        database.child(session_id).child(formatteddate).child("label").setValue(speedbreaker_l)
                        pothole_btn.setEnabled(false)
                        traffic_btn.setEnabled(false)
                        bad_road_btn.setEnabled(false)
                        sudden_change_btn.setEnabled(false)
                    }
                    else
                    {
                        sensor_label = label
//                        database.child(session_id).child(formatteddate).child("label").setValue(label)
                        speedbreaker_btn_click = 0
                        pothole_btn.setEnabled(true)
                        traffic_btn.setEnabled(true)
                        bad_road_btn.setEnabled(true)
                        sudden_change_btn.setEnabled(true)
                    }
                }
            })

            traffic_btn.setOnClickListener(object : View.OnClickListener
            {
                override fun onClick(view : View)
                {
                    traffic_btn_click = traffic_btn_click + 1
                    if (traffic_btn_click == 1)
                    {
                        sensor_label = traffic_l
//                        database.child(session_id).child(formatteddate).child("label").setValue(traffic_l)
                        pothole_btn.setEnabled(false)
                        speedbreaker_btn.setEnabled(false)
                        bad_road_btn.setEnabled(false)
                        sudden_change_btn.setEnabled(false)
                    }
                    else
                    {
                        sensor_label = label
//                        database.child(session_id).child(formatteddate).child("label").setValue(label)
                        traffic_btn_click = 0
                        pothole_btn.setEnabled(true)
                        speedbreaker_btn.setEnabled(true)
                        bad_road_btn.setEnabled(true)
                        sudden_change_btn.setEnabled(true)
                    }
                }
            })

            bad_road_btn.setOnClickListener(object : View.OnClickListener
            {
                override fun onClick(view : View)
                {
                    broad_btn_click = broad_btn_click + 1
                    if (broad_btn_click == 1)
                    {
                        sensor_label = broad_l
//                        database.child(session_id).child(formatteddate).child("label").setValue(broad_l)
                        pothole_btn.setEnabled(false)
                        traffic_btn.setEnabled(false)
                        speedbreaker_btn.setEnabled(false)
                        sudden_change_btn.setEnabled(false)
                    }
                    else
                    {
                        sensor_label = label
//                        database.child(session_id).child(formatteddate).child("label").setValue(label)
                        broad_btn_click = 0
                        pothole_btn.setEnabled(true)
                        traffic_btn.setEnabled(true)
                        speedbreaker_btn.setEnabled(true)
                        sudden_change_btn.setEnabled(true)
                    }
                }
            })

            sudden_change_btn.setOnClickListener(object : View.OnClickListener
            {
                override fun onClick(view : View)
                {
                    schange_btn_click = schange_btn_click + 1
                    if (schange_btn_click == 1)
                    {
                        sensor_label = schange_l
//                        database.child(session_id).child(formatteddate).child("label").setValue(schange_l)
                        pothole_btn.setEnabled(false)
                        traffic_btn.setEnabled(false)
                        bad_road_btn.setEnabled(false)
                        speedbreaker_btn.setEnabled(false)
                    }
                    else
                    {
                        sensor_label = label
//                        database.child(session_id).child(formatteddate).child("label").setValue(label)
                        schange_btn_click = 0
                        pothole_btn.setEnabled(true)
                        traffic_btn.setEnabled(true)
                        bad_road_btn.setEnabled(true)
                        speedbreaker_btn.setEnabled(true)
                    }
                }
            })

            if (((pothole_btn_click == 1) or (traffic_btn_click == 1) or (speedbreaker_btn_click == 1) or (schange_btn_click == 1) or (broad_btn_click == 1)))
            {
                if (pothole_btn_click == 1)
                {
                    sensor_label = pothole_l
//                    database.child(session_id).child(formatteddate).child("label").setValue(pothole_l)
                }
                else if (traffic_btn_click == 1)
                {
                    sensor_label = traffic_l
//                    database.child(session_id).child(formatteddate).child("label").setValue(traffic_l)
                }
                else if (speedbreaker_btn_click == 1)
                {
                    sensor_label = speedbreaker_l
//                    database.child(session_id).child(formatteddate).child("label").setValue(speedbreaker_l)
                }
                else if (schange_btn_click == 1)
                {
                    sensor_label = schange_l
//                    database.child(session_id).child(formatteddate).child("label").setValue(schange_l)
                }
                else if (broad_btn_click == 1)
                {
                    sensor_label = broad_l
//                    database.child(session_id).child(formatteddate).child("label").setValue(broad_l)
                }
            }
            else
            {
                sensor_label = label
//                database.child(session_id).child(formatteddate).child("label").setValue(label)
            }


            try{
                latitude = locationResult.latitude
                longitude = locationResult.longitude
                findViewById<TextView>(R.id.longt).text = longitude.toString()
                findViewById<TextView>(R.id.lat).text = latitude.toString()
//                database.child(session_id).child(formatteddate).child("latitude").setValue(latitude.toString())
//                database.child(session_id).child(formatteddate).child("longitude").setValue(longitude.toString())
            }
            catch(e: Exception){
                println("failed")
            }


            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                val accX = event.values[0]
                val temp01:Double = String.format("%.3f", accX).toDouble()
                findViewById<TextView>(R.id.acc_X).text = temp01.toString()

                val accY = event.values[1]
                val temp11:Double = String.format("%.3f", accY).toDouble()
                findViewById<TextView>(R.id.acc_Y).text = temp11.toString()

                val accZ = event.values[2]
                val temp21:Double = String.format("%.3f", accZ).toDouble()
                findViewById<TextView>(R.id.acc_Z).text = temp21.toString()
//
//                database.child(session_id).child(formatteddate).child("accelerometer-x").setValue(temp01.toString())
//                database.child(session_id).child(formatteddate).child("accelerometer-y").setValue(temp11.toString())
//                database.child(session_id).child(formatteddate).child("accelerometer-z").setValue(temp21.toString())

                accelerometer_x = temp01.toString()
                accelerometer_y = temp11.toString()
                accelerometer_z = temp21.toString()
            }

            if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
                val accX = event.values[0]
                val temp01:Double = String.format("%.3f", accX).toDouble()
                findViewById<TextView>(R.id.gyro_x).text = temp01.toString()

                val accY = event.values[1]
                val temp11:Double = String.format("%.3f", accY).toDouble()
                findViewById<TextView>(R.id.gyro_y).text = temp11.toString()

                val accZ = event.values[2]
                val temp21:Double = String.format("%.3f", accZ).toDouble()
                findViewById<TextView>(R.id.gyro_z).text = temp21.toString()

//                database.child(session_id).child(formatteddate).child("gyroscope-x").setValue(temp01.toString())
//                database.child(session_id).child(formatteddate).child("gyroscope-y").setValue(temp11.toString())
//                database.child(session_id).child(formatteddate).child("gyroscope-z").setValue(temp21.toString())

                gyroscope_x = temp01.toString()
                gyroscope_y = temp11.toString()
                gyroscope_z = temp21.toString()
            }

            myList.add(mutableListOf(formatteddate, android_id, sensor_label, latitude.toString(), longitude.toString(), accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z))
            println(myList)
        }

    }

    override fun onResume() {
        super.onResume()
        mSensorManager.registerListener(this, mAccelerometer, 20000, 500000)
        mSensorManager.registerListener(this, mGyroscope, 20000, 500000)

        // check for internet connection after every 5 seconds
        val connection_timer = Timer()
        connection_timer.scheduleAtFixedRate(object: TimerTask()
        {
            override fun run()
            {
                if (!resume && myList.size > 2 && checkforInternet(this@MainActivity))
                {
                    for (x in myList.drop(2))
                    {
                        // add in database
                        database.child(session_id).child(x[0]).child("timestamp").setValue(x[0])
                        database.child(session_id).child(x[0]).child("android-id").setValue(x[1])
                        database.child(session_id).child(x[0]).child("label").setValue(x[2])
                        database.child(session_id).child(x[0]).child("latitude").setValue(x[3])
                        database.child(session_id).child(x[0]).child("longitude").setValue(x[4])
                        database.child(session_id).child(x[0]).child("accelerometer-x").setValue(x[5])
                        database.child(session_id).child(x[0]).child("accelerometer-y").setValue(x[6])
                        database.child(session_id).child(x[0]).child("accelerometer-z").setValue(x[7])
                        database.child(session_id).child(x[0]).child("gyroscope-x").setValue(x[8])
                        database.child(session_id).child(x[0]).child("gyroscope-y").setValue(x[9])
                        database.child(session_id).child(x[0]).child("gyroscope-z").setValue(x[10])

                        myList.remove(x)
                    }
                }
            }
        },5000, 5000)
    }

    override fun onPause() {
        super.onPause()
        mSensorManager.unregisterListener(this)
    }

    // button start and stop
    fun resumeReading(view: View) {
        if (this.resume == true)
        {
            Toast.makeText(this, "Data collection has already been started!", Toast.LENGTH_SHORT).show()
        }
        else if (this.resume == false)
        {
            this.resume = true
            Toast.makeText(this, "Data collection has begun!", Toast.LENGTH_SHORT).show()
        }
    }

    fun pauseReading(view: View) {
        if (this.resume == false)
        {
            Toast.makeText(this, "Data collection has already stopped!", Toast.LENGTH_SHORT).show()
        }
        else if (this.resume == true)
        {
            this.resume = false
            Toast.makeText(this, "Data collection has stopped!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun checkforInternet(context : Context) : Boolean
    {
        // register activity with the connectivity manager service
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

        // if the android version is equal to M
        // or greater we need to use the
        // NetworkCapabilities to check what type of
        // network has the internet connection
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {

            // Returns a Network object corresponding to
            // the currently active default data network.
            val network = connectivityManager.activeNetwork ?: return false

            // Representation of the capabilities of an active network.
            val activeNetwork = connectivityManager.getNetworkCapabilities(network) ?: return false

            return when {
                // Indicates this network uses a Wi-Fi transport,
                // or WiFi has network connectivity
                activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> true

                // Indicates this network uses a Cellular transport. or
                // Cellular has network connectivity
                activeNetwork.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> true

                // else return false
                else -> false
            }
        }
        else {
            // if the android version is below M
            @Suppress("DEPRECATION") val networkInfo = connectivityManager.activeNetworkInfo ?: return false
            @Suppress("DEPRECATION")
            return networkInfo.isConnected
        }
    }

    // LOCATION STUFF
    @SuppressLint("MissingPermission")
    private fun getLastLocation() {
        if(CheckPermission()){
            if(isLocationEnabled()){

                fusedLocationProviderClient.lastLocation.addOnCompleteListener{task ->
                        getNewLocation()
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
        val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
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
        locationRequest.interval = UPDATE_INTERVAL_IN_MILLISECONDS
        locationRequest.fastestInterval = FASTEST_UPDATE_INTERVAL_IN_MILLISECONDS
        fusedLocationProviderClient!!.requestLocationUpdates(
            locationRequest, locationCallback, null
        )

    }

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(p0: LocationResult?){
            val lastLocation = p0?.lastLocation

            if (lastLocation != null) {
                locationResult = lastLocation
            }
        }

    }
}

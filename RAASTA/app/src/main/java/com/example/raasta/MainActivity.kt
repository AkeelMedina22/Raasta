package com.example.raasta


import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.pm.PackageManager.PERMISSION_GRANTED
import android.location.Location
import android.location.LocationManager
import android.os.Bundle
import android.os.Looper
import android.view.Window
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.preference.PreferenceManager
import com.google.android.gms.location.*
import com.google.android.gms.tasks.CancellationToken
import com.google.android.gms.tasks.CancellationTokenSource
import com.google.android.gms.tasks.OnTokenCanceledListener
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.osmdroid.api.IMapController
import org.osmdroid.bonuspack.routing.OSRMRoadManager
import org.osmdroid.bonuspack.routing.RoadManager
import org.osmdroid.config.Configuration
import org.osmdroid.config.Configuration.getInstance
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.mylocation.MyLocationNewOverlay


class MainActivity : AppCompatActivity(){

    // location initialization
    val PERMISSION_ID = 42
    companion object {
        private const val UPDATE_INTERVAL_IN_MILLISECONDS = 500L
        private const val FASTEST_UPDATE_INTERVAL_IN_MILLISECONDS =
            UPDATE_INTERVAL_IN_MILLISECONDS / 2
    }
    lateinit var fusedLocationProviderClient: FusedLocationProviderClient
    lateinit var locationRequest : LocationRequest
    lateinit var locationResult : Location
    private lateinit var locationCallback: LocationCallback

    var latitude : Double = 0.0
    var longitude : Double = 0.0


    var start : Location? = null

    val roadManager: RoadManager = OSRMRoadManager(this, "Raasta")
    val waypoints = ArrayList<GeoPoint>()


    // map intialization
    private lateinit var map: MapView
    private lateinit var mapController: IMapController
    lateinit var myLocationOverlay: MyLocationNewOverlay

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
       
        // restrict the thread to work only in main thread
        val policy = ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)

        locationResult = Location("current_location")
        // layout initialization
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        // Request Location permission
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PERMISSION_GRANTED
        ) {
            println("Location Permission GRANTED")
        } else {
            println("Location Permission DENIED")
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                1
            )
        }

        // MAP
        Configuration.getInstance().setUserAgentValue(this.getPackageName())
        getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this))

        setContentView(R.layout.activity_main)

        map = findViewById<MapView>(R.id.map)
        map.setTileSource(TileSourceFactory.MAPNIK)
        map.isClickable = true
        map.setMultiTouchControls(true)

        // get current location
        fusedLocationProviderClient = LocationServices.getFusedLocationProviderClient(this)
        GlobalScope.launch{
            locationResult = getCurrentLocation()
        }

        println(locationResult.latitude)
       
        // hard-coded location point is used to center the map
        val mapController = map.controller
        val startPoint = GeoPoint(24.956, 67.052)
        mapController.setZoom(18.5)
        mapController.setCenter(startPoint)
        
        // placing marker at the starting point which is the hard coded location
        val startMarker = Marker(map)
        startMarker.position = startPoint
        startMarker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
        map.overlays.add(startMarker)

        // route generation
        val roadManager: RoadManager = OSRMRoadManager(this, "Raasta")

        val waypoints = ArrayList<GeoPoint>()
        waypoints.add(startPoint)
        // hard coding the end point location which is habib university
        val endPoint = GeoPoint(24.905865, 67.138235)
        waypoints.add(endPoint)

        val road = roadManager.getRoad(waypoints)
        // making the route
        val roadOverlay: Polyline = RoadManager.buildRoadOverlay(road)
        map.getOverlays().add(roadOverlay)

        // add pin at habib univeristy
        val endMarker = Marker(map)
        endMarker.position = endPoint
        endMarker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
        map.overlays.add(endMarker)
        
        // drop a pin
        val mReceive: MapEventsReceiver = object : MapEventsReceiver {
            override fun singleTapConfirmedHelper(p: GeoPoint): Boolean {
                return false
            }

            override fun longPressHelper(p: GeoPoint): Boolean {
                // write your code here
                // did not drop pin yet, just shows the location points rn
                val str = p.latitude.toString() + " - " + p.longitude.toString()
                Toast.makeText(baseContext, str ,Toast.LENGTH_LONG).show()
                return false
            }
        }
        val OverlayEvents = MapEventsOverlay(mReceive)
        map.overlays.add(OverlayEvents)
        
        // adding marker at specific points of the route
//        val nodeIcon = resources.getDrawable(R.drawable.marker_node)
//        for (i in road.mNodes.indices) {
//            val node = road.mNodes[i]
//            val nodeMarker = Marker(map)
//            nodeMarker.position = node.mLocation
//            nodeMarker.icon = nodeIcon
//            nodeMarker.title = "Step $i"
//
//            nodeMarker.setSnippet(node.mInstructions)
//            nodeMarker.setSubDescription(Road.getLengthDurationText(this, node.mLength, node.mDuration))
//            val icon = resources.getDrawable(R.drawable.ic_continue)
//            nodeMarker.image = icon
//            map.overlays.add(nodeMarker)
//        }
//
//        // search for points of interests
//        val poiProvider = NominatimPOIProvider("OSMBonusPackTutoUserAgent")
//        val pois = poiProvider.getPOICloseTo(startPoint, "cinema", 50, 0.1)
//
//        val poiMarkers = FolderOverlay(this)
//        map.overlays.add(poiMarkers)
//
//        val poiIcon = resources.getDrawable(R.drawable.marker_poi_default)
//        for (poi in pois) {
//            val poiMarker = Marker(map)
//            poiMarker.title = poi.mType
//            poiMarker.snippet = poi.mDescription
//            poiMarker.position = poi.mLocation
//            poiMarker.icon = poiIcon
//            if (poi.mThumbnail != null) {
//                poiMarker.setImage(BitmapDrawable(poi.mThumbnail))
//            }
//            poiMarkers.add(poiMarker)
//        }

   
//        map.overlays.add(myLocationOverlay)

//        map.invalidate()

        //Offline maps:
        //map.setUseDataConnection(true);

    }

    override fun onResume()
    {
        super.onResume()
        map.onResume()
    }

    override fun onPause()
    {
        super.onPause()
        map.onPause()

    }

    suspend fun getCurrentLocation() : Location
    {
        val def = CompletableDeferred<Location>()
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PERMISSION_GRANTED)
        {
            fusedLocationProviderClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, object : CancellationToken() {
                override fun onCanceledRequested(listener: OnTokenCanceledListener) = CancellationTokenSource().token
                override fun isCancellationRequested() = false }).addOnSuccessListener{
                    if (it != null)
                    {
                        def.complete(it)
                    }
                }
        }
        return def.await()

    }
//    private fun getCurrentLocation()
//    {
//        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PERMISSION_GRANTED)
//        {
//            fusedLocationProviderClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, object : CancellationToken() {
//                override fun onCanceledRequested(listener: OnTokenCanceledListener) = CancellationTokenSource().token
//                override fun isCancellationRequested() = false }).addOnSuccessListener{
//                    if (it != null)
//                    {
//                        showMap(it)
//                    }
//                }
//        }
//    }

    private fun showMap(location : Location)
    {
        println(location.longitude)
        println(location.latitude)
        val mapController = map.controller
        val startPoint = GeoPoint(location.latitude, location.longitude)
        mapController.setZoom(18.5)
        mapController.setCenter(startPoint)

        val startMarker = Marker(map)
        startMarker.position = startPoint
        startMarker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
        map.overlays.add(startMarker)


    }

}

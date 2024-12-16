from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def clustering_model(self, image):
        # convert image to 2d array
        image_2d = image.reshape(-1, 3)
    
        #perform k-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10).fit(image_2d)
        
        return kmeans
    

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image[:int(image.shape[0]//2), :]

        # get clustering model
        kmeans = self.clustering_model(top_half)
    
        # get cluster labels
        labels = kmeans.labels_

        # reshape to original image shape
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])

        # get the player color
        corner_cluster = clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    

    def assign(self, frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1).fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0] 
        self.team_colors[2] = kmeans.cluster_centers_[1]   

    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        if player_id == 118:
            team_id = 1
        elif player_id == 262:
            team_id = 2

        self.player_team_dict[player_id] = team_id

        return team_id
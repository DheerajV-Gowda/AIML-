import pygame
import math
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vacuum Cleaner Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)
BROWN = (139, 69, 19)

# Vacuum cleaner class
class VacuumCleaner:
    def __init__(self, x, y, shape_type):
        self.x = x
        self.y = y
        self.shape_type = shape_type
        self.angle = 0
        self.speed = 0
        self.rotation_speed = 3
        self.max_speed = 5
        self.cleaning = False
        self.docked = False
        self.battery = 100
        self.dust_collected = 0
        self.size = 30
        
    def draw(self, screen):
        if self.shape_type == 1:  # Circular shape
            pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), self.size)
            pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.size, 2)
            # Direction indicator
            end_x = self.x + math.cos(math.radians(self.angle)) * self.size
            end_y = self.y - math.sin(math.radians(self.angle)) * self.size
            pygame.draw.line(screen, RED, (self.x, self.y), (end_x, end_y), 3)
            
        elif self.shape_type == 2:  # Rectangular shape
            points = []
            for angle in [0, 90, 180, 270]:
                rad_angle = math.radians(self.angle + angle)
                px = self.x + math.cos(rad_angle) * self.size * 1.2
                py = self.y - math.sin(rad_angle) * self.size * 0.8
                points.append((px, py))
            pygame.draw.polygon(screen, GREEN, points)
            pygame.draw.polygon(screen, BLACK, points, 2)
            # Direction indicator
            end_x = self.x + math.cos(math.radians(self.angle)) * self.size
            end_y = self.y - math.sin(math.radians(self.angle)) * self.size
            pygame.draw.line(screen, RED, (self.x, self.y), (end_x, end_y), 3)
            
        elif self.shape_type == 3:  # Triangular shape
            points = []
            for angle in [0, 120, 240]:
                rad_angle = math.radians(self.angle + angle)
                px = self.x + math.cos(rad_angle) * self.size
                py = self.y - math.sin(rad_angle) * self.size
                points.append((px, py))
            pygame.draw.polygon(screen, YELLOW, points)
            pygame.draw.polygon(screen, BLACK, points, 2)
            # Direction indicator (front of triangle)
            end_x = self.x + math.cos(math.radians(self.angle)) * self.size * 1.5
            end_y = self.y - math.sin(math.radians(self.angle)) * self.size * 1.5
            pygame.draw.line(screen, RED, (self.x, self.y), (end_x, end_y), 3)
            
        elif self.shape_type == 4:  # D-shaped (half circle)
            # Draw the semicircle
            rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)
            pygame.draw.arc(screen, RED, rect, math.radians(self.angle - 90), math.radians(self.angle + 90), self.size)
            # Draw the flat side
            start_x = self.x + math.cos(math.radians(self.angle + 90)) * self.size
            start_y = self.y - math.sin(math.radians(self.angle + 90)) * self.size
            end_x = self.x + math.cos(math.radians(self.angle - 90)) * self.size
            end_y = self.y - math.sin(math.radians(self.angle - 90)) * self.size
            pygame.draw.line(screen, RED, (start_x, start_y), (end_x, end_y), 3)
            # Direction indicator
            end_x = self.x + math.cos(math.radians(self.angle)) * self.size
            end_y = self.y - math.sin(math.radians(self.angle)) * self.size
            pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 3)
        
        # Draw cleaning effect if cleaning
        if self.cleaning:
            if self.shape_type == 1:  # Circular - 360 degree cleaning
                pygame.draw.circle(screen, LIGHT_BLUE, (int(self.x), int(self.y)), self.size + 5, 2)
            elif self.shape_type == 2:  # Rectangular - front focused
                end_x = self.x + math.cos(math.radians(self.angle)) * (self.size + 15)
                end_y = self.y - math.sin(math.radians(self.angle)) * (self.size + 15)
                pygame.draw.line(screen, LIGHT_BLUE, (self.x, self.y), (end_x, end_y), 5)
            elif self.shape_type == 3:  # Triangular - wide front
                angle1 = self.angle - 30
                angle2 = self.angle + 30
                end_x1 = self.x + math.cos(math.radians(angle1)) * (self.size + 15)
                end_y1 = self.y - math.sin(math.radians(angle1)) * (self.size + 15)
                end_x2 = self.x + math.cos(math.radians(angle2)) * (self.size + 15)
                end_y2 = self.y - math.sin(math.radians(angle2)) * (self.size + 15)
                pygame.draw.polygon(screen, LIGHT_BLUE, [(self.x, self.y), (end_x1, end_y1), (end_x2, end_y2)], 2)
            elif self.shape_type == 4:  # D-shaped - front focused
                end_x = self.x + math.cos(math.radians(self.angle)) * (self.size + 15)
                end_y = self.y - math.sin(math.radians(self.angle)) * (self.size + 15)
                pygame.draw.line(screen, LIGHT_BLUE, (self.x, self.y), (end_x, end_y), 5)
    
    def move(self):
        if not self.docked:
            self.x += self.speed * math.cos(math.radians(self.angle))
            self.y -= self.speed * math.sin(math.radians(self.angle))
            
            # Boundary checking
            self.x = max(self.size, min(WIDTH - self.size, self.x))
            self.y = max(self.size, min(HEIGHT - self.size, self.y))
            
            # Battery consumption
            if self.speed > 0 or self.cleaning:
                self.battery -= 0.05
                if self.battery <= 0:
                    self.battery = 0
                    self.speed = 0
                    self.cleaning = False
        else:
            # Charging when docked
            self.battery += 0.1
            if self.battery >= 100:
                self.battery = 100
    
    def start(self):
        if self.battery > 5 and not self.docked:
            self.speed = self.max_speed / 2
            self.cleaning = True
    
    def stop(self):
        self.speed = 0
        self.cleaning = False
    
    def left(self):
        self.angle = (self.angle + self.rotation_speed) % 360
    
    def right(self):
        self.angle = (self.angle - self.rotation_speed) % 360
    
    def dock(self):
        # Check if near docking station
        if math.sqrt((self.x - docking_station.x)**2 + (self.y - docking_station.y)**2) < 60:
            self.docked = True
            self.speed = 0
            self.cleaning = False
        else:
            self.docked = False
    
    def undock(self):
        self.docked = False

# Docking station class
class DockingStation:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 40
    
    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, (self.x - self.size/2, self.y - self.size/2, self.size, self.size))
        pygame.draw.rect(screen, BLACK, (self.x - self.size/2, self.y - self.size/2, self.size, self.size), 2)
        # Draw charging contacts
        pygame.draw.rect(screen, YELLOW, (self.x - 15, self.y - 20, 10, 5))
        pygame.draw.rect(screen, YELLOW, (self.x + 5, self.y - 20, 10, 5))

# Dust particle class
class Dust:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 3
        self.collected = False
    
    def draw(self, screen):
        if not self.collected:
            pygame.draw.circle(screen, BROWN, (int(self.x), int(self.y)), self.size)

# Create vacuum cleaner with initial shape
vacuum = VacuumCleaner(WIDTH // 2, HEIGHT // 2, 1)
docking_station = DockingStation(100, 100)

# Create dust particles
dust_particles = []
for _ in range(50):
    dust_particles.append(Dust(
        random.randint(50, WIDTH - 50),
        random.randint(50, HEIGHT - 50)
    ))

# Font for text
font = pygame.font.SysFont(None, 24)

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                vacuum.speed = min(vacuum.speed + 1, vacuum.max_speed)
            elif event.key == pygame.K_DOWN:
                vacuum.speed = max(vacuum.speed - 1, -vacuum.max_speed/2)
            elif event.key == pygame.K_LEFT:
                vacuum.left()
            elif event.key == pygame.K_RIGHT:
                vacuum.right()
            elif event.key == pygame.K_SPACE:
                vacuum.cleaning = not vacuum.cleaning
            elif event.key == pygame.K_d:
                vacuum.dock()
            elif event.key == pygame.K_u:
                vacuum.undock()
            elif event.key == pygame.K_s:
                if vacuum.speed == 0:
                    vacuum.start()
                else:
                    vacuum.stop()
            elif event.key == pygame.K_1:
                vacuum.shape_type = 1
            elif event.key == pygame.K_2:
                vacuum.shape_type = 2
            elif event.key == pygame.K_3:
                vacuum.shape_type = 3
            elif event.key == pygame.K_4:
                vacuum.shape_type = 4
    
    # Move vacuum
    vacuum.move()
    
    # Check for dust collection
    if vacuum.cleaning:
        for dust in dust_particles:
            if not dust.collected:
                dist = math.sqrt((dust.x - vacuum.x)**2 + (dust.y - vacuum.y)**2)
                if dist < vacuum.size + 10:
                    dust.collected = True
                    vacuum.dust_collected += 1
    
    # Draw everything
    screen.fill(WHITE)
    
    # Draw floor pattern
    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, (220, 220, 220), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, (220, 220, 220), (0, y), (WIDTH, y))
    
    # Draw dust particles
    for dust in dust_particles:
        dust.draw(screen)
    
    # Draw docking station
    docking_station.draw(screen)
    
    # Draw vacuum cleaner
    vacuum.draw(screen)
    
    # Draw UI panel
    pygame.draw.rect(screen, (240, 240, 240), (0, 0, WIDTH, 30))
    pygame.draw.line(screen, BLACK, (0, 30), (WIDTH, 30))
    
    # Draw battery indicator
    battery_width = 100
    pygame.draw.rect(screen, BLACK, (10, 10, battery_width, 15), 1)
    fill_width = int((vacuum.battery / 100) * (battery_width - 4))
    battery_color = GREEN if vacuum.battery > 30 else RED
    pygame.draw.rect(screen, battery_color, (12, 12, fill_width, 11))
    
    # Draw dust collected indicator
    dust_text = font.render(f"Dust: {vacuum.dust_collected}", True, BLACK)
    screen.blit(dust_text, (120, 10))
    
    # Draw status text
    status = "Docked" if vacuum.docked else "Cleaning" if vacuum.cleaning else "Stopped"
    status_text = font.render(f"Status: {status}", True, BLACK)
    screen.blit(status_text, (220, 10))
    
    # Draw shape info
    shape_names = ["Circular", "Rectangular", "Triangular", "D-Shaped"]
    shape_text = font.render(f"Shape: {shape_names[vacuum.shape_type-1]}", True, BLACK)
    screen.blit(shape_text, (350, 10))
    
    # Draw controls help
    controls_text = font.render("Controls: Arrows=Move, Space=Clean, D=Dock, U=Undock, S=Start/Stop, 1-4=Change Shape", True, BLACK)
    screen.blit(controls_text, (500, 10))
    
    # Draw shape advantages
    advantages = [
        "Circular: 360° cleaning, maneuverable in tight spaces",
        "Rectangular: Efficient for straight paths and edges",
        "Triangular: Better corner cleaning, wide front coverage",
        "D-Shaped: Combines straight edge cleaning with rounded maneuverability"
    ]
    advantage_text = font.render(advantages[vacuum.shape_type-1], True, BLUE)
    screen.blit(advantage_text, (10, HEIGHT - 30))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()